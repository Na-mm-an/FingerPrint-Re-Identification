"""
train.py

Trains the fingerprint embedding network with batch-hard triplet loss on
SOCOFing, using a subject-level train/val/test split (see socofing_index.py)
so that val/test subjects' identities are never seen during training -- this
is what makes the evaluation meaningful for an open-set biometric system.

CHANGES vs. the first version (after observing embedding collapse -- loss
flatlining at ~margin with active_triplets stuck at 100% for 30 straight
epochs, meaning the model mapped every fingerprint to nearly the same point
instead of learning to discriminate identities):

  1. Lower default LR (1e-4 instead of 1e-3) -- batch-hard triplet loss from
     a randomly-initialized network is prone to collapsing early when the
     LR is too aggressive.
  2. Linear warmup for the first --warmup_steps optimizer steps, before the
     cosine decay schedule takes over. Starting from a very small LR gives
     the network a chance to find real discriminative structure before
     taking large steps.
  3. Gradient norm clipping (--grad_clip), which further guards against the
     large, collapse-inducing updates that can happen in the first few
     batches of triplet-loss training.
  4. An embedding-spread check every epoch (reuses the same idea as
     diagnose_embeddings.py) that prints a warning the moment collapse
     starts happening, instead of only finding out after wasting a full
     30-epoch run.

CHANGES vs. the second version (after observing that collapse persisted
even with a tiny peak LR of ~1e-5 and well past the warmup window, getting
monotonically worse every epoch regardless of the cosine-decay schedule):

  5. Switched from plain Adam (with L2-penalty weight decay applied
     uniformly to every parameter) to AdamW with weight decay EXCLUDED for
     BatchNorm affine parameters (gamma/beta) and all biases. Because every
     conv in ConvBlock uses bias=False, BatchNorm's gamma is the only thing
     scaling the input-dependent signal in the stem/stages; decaying gamma
     toward zero collapses every channel toward its (input-independent)
     beta, which manifests as exactly the LR-independent, steady embedding
     collapse that was observed. Adam's per-parameter adaptive step size
     makes this worse than it would be under SGD, since the weight-decay
     pull on a low-gradient parameter like gamma can dominate its update
     even at small nominal weight_decay values (see Loshchilov & Hutter,
     "Decoupled Weight Decay Regularization").

Usage:
    python train.py --data_root /path/to/SOCOFing --epochs 30 \
        --p 16 --k 4 --lr 1e-4 --warmup_steps 500 --out_dir ./checkpoints

Note on --data_root: it should be the folder that directly contains "Real"
and "Altered" (i.e. .../SOCOFing).
"""

from __future__ import annotations

import argparse
import os
import time
import math

import torch
from torch.utils.data import DataLoader

from socofing_index import index_socofing, group_by_subject, split_subjects_train_val_test
from dataset import SocofingReIDDataset, PKSampler, build_label_map
from model import FingerprintEmbeddingNet
from losses import BatchHardTripletLoss, pairwise_euclidean_distances


def build_splits(records, val_frac=0.15, test_frac=0.15, seed=42):
    grouped = group_by_subject(records)
    train_ids, val_ids, test_ids = split_subjects_train_val_test(
        list(grouped.keys()), val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    train_records = [r for r in records if r.subject_id in train_ids]
    val_records = [r for r in records if r.subject_id in val_ids]
    test_records = [r for r in records if r.subject_id in test_ids]
    return train_records, val_records, test_records


def evaluate_val_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = model(images)
            loss, _ = loss_fn(embeddings, labels)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


@torch.no_grad()
def check_embedding_spread(model, loader, device, max_batches=1):
    """Quick collapse check: embeds a batch and reports the spread of
    pairwise distances. Returns (mean_dist, std_dist). If std is tiny and
    mean is near 0, the model is collapsing every input to ~the same point.
    """
    model.eval()
    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        embeddings = model(images)
        dist = pairwise_euclidean_distances(embeddings)
        n = dist.size(0)
        off_diag = dist[~torch.eye(n, dtype=torch.bool, device=device)]
        model.train()
        return off_diag.mean().item(), off_diag.std().item()
    model.train()
    return None, None


def make_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """LambdaLR: linear warmup from ~0 to base_lr over warmup_steps, then
    cosine decay to 0 over the remaining steps. Operates per-OPTIMIZER-STEP
    (not per-epoch), since a fixed number of warmup epochs can be too coarse
    -- a bad first few hundred steps is enough to trigger collapse.
    """
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with weight decay excluded for BatchNorm affine params and all
    biases.

    Every ConvBlock conv uses bias=False, so BatchNorm's gamma is the only
    thing scaling the input-dependent signal through the stem/stages, and
    beta is the only additive term. Applying weight decay to gamma pulls it
    toward zero, which pushes every channel's output toward the constant
    beta -- i.e. toward an input-INDEPENDENT embedding. That is a direct
    mechanism for embedding collapse that has nothing to do with the LR
    schedule, so it must be excluded here rather than fixed by tuning --lr.

    Params are grouped by ndim: BatchNorm weight/bias and any Linear/Conv
    bias are 1-D tensors, so `param.ndim <= 1` cleanly separates them from
    the 2-D+ conv/linear weight tensors that should still be decayed.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                         help="Path to the folder containing SOCOFing's Real/ and Altered/ dirs")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--p", type=int, default=16, help="distinct subjects per batch")
    parser.add_argument("--k", type=int, default=4, help="images per subject per batch")
    parser.add_argument("--iterations_per_epoch", type=int, default=200)
    parser.add_argument("--val_iterations", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Lowered from 1e-3: high LR was contributing to embedding collapse")
    parser.add_argument("--warmup_steps", type=int, default=500,
                         help="Linear LR warmup steps before cosine decay begins")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                         help="Max gradient norm; set to 0 to disable clipping")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help="Applied via AdamW, and only to conv/linear weight tensors -- "
                              "BatchNorm gamma/beta and biases are excluded (see build_optimizer)")
    parser.add_argument("--include_altered_levels", type=str, default="Easy,Medium,Hard")
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--collapse_std_threshold", type=float, default=0.02,
                         help="Warn if embedding pairwise-distance std drops below this")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)

    levels = [s for s in args.include_altered_levels.split(",") if s]
    records = index_socofing(args.data_root, include_altered_levels=levels)
    train_records, val_records, test_records = build_splits(records, seed=args.seed)
    print(f"Records -> train: {len(train_records)}  val: {len(val_records)}  test: {len(test_records)}")

    # IMPORTANT: label maps must be built PER SPLIT since train/val/test
    # contain disjoint subject id sets (that's the whole point of a subject-
    # level split for open-set evaluation). Each split gets its own
    # contiguous label indices for triplet mining purposes.
    train_label_map = build_label_map(train_records)
    val_label_map = build_label_map(val_records)

    train_ds = SocofingReIDDataset(train_records, train_label_map, image_size=args.image_size, train=True)
    val_ds = SocofingReIDDataset(val_records, val_label_map, image_size=args.image_size, train=False)

    train_sampler = PKSampler(train_ds, p=args.p, k=args.k, iterations=args.iterations_per_epoch)
    val_sampler = PKSampler(val_ds, p=min(args.p, len(val_label_map)), k=args.k, iterations=args.val_iterations)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=args.num_workers)

    model = FingerprintEmbeddingNet(embedding_dim=args.embedding_dim).to(device)
    loss_fn = BatchHardTripletLoss(margin=args.margin)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * args.iterations_per_epoch
    scheduler = make_warmup_cosine_scheduler(optimizer, args.warmup_steps, total_steps)

    best_val_loss = float("inf")
    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_active_frac = 0.0, 0.0
        t0 = time.time()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            loss, diagnostics = loss_fn(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            epoch_active_frac += diagnostics["fraction_active_triplets"]

        n_batches = len(train_loader)
        train_loss = epoch_loss / n_batches
        active_frac = epoch_active_frac / n_batches
        val_loss = evaluate_val_loss(model, val_loader, loss_fn, device)
        spread_mean, spread_std = check_embedding_spread(model, val_loader, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss {train_loss:.4f} "
            f"| val_loss {val_loss:.4f} | active_triplets {active_frac:.2%} "
            f"| embed_dist mean={spread_mean:.4f} std={spread_std:.4f} "
            f"| lr {current_lr:.2e} | {elapsed:.1f}s"
        )

        if spread_std is not None and spread_std < args.collapse_std_threshold:
            print(
                f"  !! WARNING: embedding pairwise-distance std ({spread_std:.4f}) is below "
                f"threshold ({args.collapse_std_threshold}) -- possible embedding collapse. "
                f"If this persists past the warmup period ({args.warmup_steps} steps, "
                f"currently at step {global_step}), consider lowering --lr further, and double "
                f"check --weight_decay isn't being applied to BatchNorm/bias params."
            )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(), "embedding_dim": args.embedding_dim,
                 "image_size": args.image_size, "epoch": epoch, "val_loss": val_loss},
                os.path.join(args.out_dir, "best_model.pt"),
            )

    torch.save(
        {"model_state": model.state_dict(), "embedding_dim": args.embedding_dim,
         "image_size": args.image_size, "epoch": args.epochs, "val_loss": val_loss},
        os.path.join(args.out_dir, "final_model.pt"),
    )
    print(f"Done. Best val_loss={best_val_loss:.4f}. Checkpoints saved to {args.out_dir}")


if __name__ == "__main__":
    main()
