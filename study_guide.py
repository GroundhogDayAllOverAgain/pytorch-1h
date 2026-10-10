# =============================================================================
# STUDY GUIDE: Distributed Data Parallel (DDP) training in PyTorch
# =============================================================================
# What this script does:
#   Trains a tiny neural network on a tiny 2-feature, 2-class toy dataset,
#   using DDP so that the SAME training job can run across multiple GPUs.
#
# Core idea of DDP (data parallelism):
#   - Each GPU gets its own process and its own full copy of the model.
#   - Each process sees a DIFFERENT slice of every batch of data.
#   - After loss.backward(), DDP averages the gradients across all processes
#     (an "all-reduce"), so every model copy gets the identical update and
#     all copies stay in sync.
#
# How to run (launcher sets the environment variables this script reads):
#   torchrun --nproc_per_node=2 DDP-script-torchrun.py
#   (torchrun starts 2 processes and sets WORLD_SIZE, RANK, LOCAL_RANK,
#    MASTER_ADDR, MASTER_PORT for each of them.)
#
# Lines marked "NEW" show what was added to turn a normal single-GPU
# training script into a DDP script.
# =============================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NEW imports:
import os        # to read/write environment variables (torchrun communicates through these)
import platform  # to detect the operating system (Windows needs a different backend)
from torch.utils.data.distributed import DistributedSampler  # splits a dataset across processes
from torch.nn.parallel import DistributedDataParallel as DDP  # wrapper that syncs gradients
from torch.distributed import init_process_group, destroy_process_group  # start / stop multi-process communication


# -----------------------------------------------------------------------------
# NEW: function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
# -----------------------------------------------------------------------------
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID (0, 1, 2, ... world_size-1). Typically one
              process per GPU, so rank also doubles as the GPU index here.
        world_size: total number of processes in the group (e.g., 2 for 2 GPUs)
    """
    # MASTER_ADDR / MASTER_PORT tell all processes where to find each other
    # for the initial "handshake". The "master" is rank 0's machine.
    # torchrun normally sets these for us; we only fall back to defaults
    # (single machine, arbitrary free port) if they're not already defined,
    # e.g. when running the script without torchrun.
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # initialize process group
    # The "backend" is the library that actually moves tensors between
    # processes (used for gradient averaging, etc.).
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        # (fastest option for GPU-to-GPU communication; Linux only)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Pin this process to its own GPU. After this, "cuda" without an index
    # means GPU number `rank` for this process. Without this, every process
    # might default to GPU 0 and fight over it.
    torch.cuda.set_device(rank)


# -----------------------------------------------------------------------------
# Dataset: the standard PyTorch pattern for wrapping tensors
# -----------------------------------------------------------------------------
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X   # tensor of shape (num_examples, num_features)
        self.labels = y     # tensor of shape (num_examples,)

    def __getitem__(self, index):
        # Return ONE (input, label) pair. The DataLoader calls this repeatedly
        # and stacks the results into batches.
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        # Total number of examples. The sampler/DataLoader use this to know
        # how many indices exist.
        return self.labels.shape[0]


# -----------------------------------------------------------------------------
# Model: a small multilayer perceptron (MLP). Nothing DDP-specific in here;
# DDP works by wrapping an ordinary model later on.
# -----------------------------------------------------------------------------
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer: num_inputs -> 30 units
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer: 30 -> 20 units
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer: 20 -> num_outputs (one score per class)
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        # Returns "logits": raw, unnormalized class scores. We do NOT apply
        # softmax here because F.cross_entropy does that internally.
        logits = self.layers(x)
        return logits


# -----------------------------------------------------------------------------
# Data preparation: builds the datasets and DataLoaders
# -----------------------------------------------------------------------------
def prepare_dataset():
    # 5 training points: the first 3 belong to class 0, the last 2 to class 1.
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    # 2 test points, one per class
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # Why this is needed: with more GPUs than data, some processes would get
    # too few samples to form even one full batch (see the ZeroDivisionError
    # handling in main()). This block makes `factor` noisy copies of the data.
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,       # NOTE: this is the PER-GPU batch size. With 2 GPUs,
                            # the effective (global) batch size is 2 x 2 = 4.
        shuffle=False,      # NEW: False because of DistributedSampler below
                            # (the sampler handles shuffling itself; you can't
                            # use both shuffle=True and a sampler)
        pin_memory=True,    # allocates batches in page-locked memory for faster CPU->GPU copies
        drop_last=True,     # discard a final incomplete batch so every batch has the same size
        # NEW: chunk batches across GPUs without overlapping samples:
        # DistributedSampler gives each process (rank) its own disjoint subset
        # of the dataset indices. It reads rank and world_size from the
        # process group, which is why ddp_setup() must run BEFORE this.
        # Note: if the dataset size isn't divisible by world_size, it pads by
        # repeating a few samples so every rank gets the same count.
        sampler=DistributedSampler(train_ds)  # NEW
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        # No DistributedSampler here: every process evaluates the FULL test
        # set on its own. That's redundant but simple for a toy example.
    )
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# NEW: wrapper
# Everything that used to be top-level training code is now inside main(),
# because each GPU process will run this function independently.
# -----------------------------------------------------------------------------
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # NEW: initialize process groups

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)  # move the model's parameters to this process's GPU (cuda:rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)  # plain SGD, learning rate 0.5

    # NEW: wrap model with DDP
    # On construction, DDP broadcasts rank 0's weights to all other ranks so
    # every replica starts identical. During backward(), it hooks into
    # gradient computation and averages gradients across all processes.
    # The optimizer was created BEFORE wrapping; that's fine because DDP
    # shares the same underlying parameters.
    model = DDP(model, device_ids=[rank])
    # the core model is now accessible as model.module

    for epoch in range(num_epochs):
        # NEW: Set sampler to ensure each epoch has a different shuffle order
        # DistributedSampler shuffles using (seed + epoch). If you forget this
        # call, every epoch would use the SAME order. All ranks use the same
        # seed so their shuffles stay consistent (no overlap between ranks).
        train_loader.sampler.set_epoch(epoch)

        model.train()  # training mode (matters for dropout/batchnorm; harmless here)
        for features, labels in train_loader:

            features, labels = features.to(rank), labels.to(rank)  # New: use rank
            # forward pass on this process's own mini-batch
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()  # clear gradients from the previous step
            loss.backward()        # compute gradients; DDP all-reduces (averages) them across GPUs here
            optimizer.step()       # update weights; since gradients are identical on every GPU,
                                   # all model replicas remain identical

            # LOGGING
            # Every rank prints its own line, so with 2 GPUs you'll see
            # [GPU0] and [GPU1] lines interleaved. The printed loss is the
            # LOCAL loss on this rank's batch (before averaging).
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()  # switch to evaluation mode

    try:
        # Evaluate on training and test data (each rank does this independently).
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    ####################################################
    # NEW:
    # When does ZeroDivisionError happen? compute_accuracy divides by
    # total_examples. If there are more GPUs than the tiny dataset can feed
    # (e.g., 8 GPUs: each rank gets 1 training sample, but batch_size=2 with
    # drop_last=True means ZERO batches), total_examples stays 0 and the
    # division fails. This handler re-raises with a friendlier explanation.
    # (Note: the original message referred to "lines 103 to 107"; that's the
    # commented-out dataset-enlarging block in prepare_dataset().)
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "torchrun --nproc_per_node=2 DDP-script-torchrun.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the dataset-enlarging code in prepare_dataset()."
        )
    ####################################################

    destroy_process_group()  # NEW: cleanly exit distributed mode
    # (shuts down communication and frees resources; avoids warnings/hangs on exit)


# -----------------------------------------------------------------------------
# Evaluation helper: computes accuracy = (# correct predictions) / (# examples)
# -----------------------------------------------------------------------------
def compute_accuracy(model, dataloader, device):
    model = model.eval()  # make sure we're in eval mode
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        # no_grad: skip building the autograd graph; saves memory and time
        # since we're only predicting, not training.
        with torch.no_grad():
            logits = model(features)
        # The predicted class is the index of the largest logit in each row
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions         # boolean tensor: True where correct
        correct += torch.sum(compare)           # count the Trues in this batch
        total_examples += len(compare)          # count examples seen so far
    # .item() converts the 0-dim tensor to a plain Python float.
    # Raises ZeroDivisionError if the loader yielded no batches.
    return (correct / total_examples).item()


# -----------------------------------------------------------------------------
# Entry point. torchrun launches one copy of this whole script per GPU, and
# each copy runs this block with its own environment variables.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # NEW: Use environment variables set by torchrun if available, otherwise default to single-process.

    # WORLD_SIZE = total number of processes across all machines
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    # LOCAL_RANK = this process's index on its own machine (used as the GPU index)
    # RANK       = this process's global index across all machines
    # On a single machine they're identical. On multiple machines, LOCAL_RANK
    # is the correct choice for picking a GPU, so it's checked first.
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    # Only print on rank 0 to avoid duplicate prints from each GPU process
    if rank == 0:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Number of GPUs available:", torch.cuda.device_count())

    # Same seed in every process => identical initial weights and consistent
    # sampler shuffling across ranks. (DDP also broadcasts weights from rank 0
    # as a safeguard.)
    torch.manual_seed(123)
    num_epochs = 3
    main(rank, world_size, num_epochs)