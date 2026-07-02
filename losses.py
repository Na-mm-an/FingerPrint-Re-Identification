"""
losses.py

Batch-hard triplet loss (Hermans et al., "In Defense of the Triplet Loss for
Person Re-Identification", 2017).

Why batch-hard instead of pre-picking (anchor, positive, negative) triplets
ahead of time: most randomly-picked triplets are already "easy" (the
negative is already far away), which contributes ~zero gradient and wastes
compute. Batch-hard mining looks at every embedding in a P-K batch and, for
each anchor, picks the HARDEST positive (same subject, most distant) and the
HARDEST negative (different subject, closest) actually present in that
batch. This is the standard, effective version of triplet loss and is why
the PKSampler in dataset.py exists -- it guarantees enough same-subject and
different-subject examples per batch for this mining to work.

CHANGE (after observing embedding collapse that got WORSE and plateaued at a
suspiciously tiny, near-constant pairwise-distance std of ~2e-4-4e-4, right
around float32's precision floor): pairwise_euclidean_distances previously
computed dist_sq = ||a||^2 + ||b||^2 - 2*a.b directly and then did
dist_sq.clamp(min=0.0) before sqrt. For L2-normalized embeddings this
collapses to dist_sq = 2 - 2*cos_sim, which is a subtraction of two O(1)
numbers whose difference becomes tiny once embeddings converge -- classic
catastrophic cancellation. Once the true squared distance drops below
float32's precision floor for that subtraction (~1e-6-1e-7 here), the
computed value is essentially noise and sometimes lands negative, which
clamp(min=0.0) zeroes out. torch.clamp has EXACTLY ZERO gradient in the
clamped region, so once floating-point noise pushes a pair's distance to
the clamp floor, that pair stops receiving ANY repulsive gradient -- while
attraction on positive pairs is unaffected. That's a one-way ratchet that
locks in and entrenches collapse once it starts, and explains the plateau
at a fixed tiny std rather than the loss continuing to move at all.

Fix: clamp to a much larger, deliberately-chosen epsilon (not exactly 0),
so there's always a small but well-conditioned nonzero floor with a
meaningful (non-clamped, non-exploding) gradient, and so "very close"
points still produce a small but real and correctly-signed repulsive
gradient instead of exactly zero. This does not fix whatever is causing the
INITIAL rapid collapse (that appears to happen before distances are ever
small enough for this numerical issue to matter) -- it only prevents this
specific mechanism from being a self-reinforcing trap once collapse starts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Deliberately much larger than the previous 1e-12. 1e-12 is far below
# float32 precision (~1e-7 relative) for the O(1)-scale subtraction used
# below, so it does nothing to prevent the clamp from biting on noise --
# it only avoided a literal NaN at dist_sq == 0 exactly. 1e-6 keeps the
# sqrt gradient (1 / (2*sqrt(dist_sq + eps))) from exploding while still
# being small enough not to meaningfully distort real (non-degenerate)
# distances.
_DIST_SQ_EPS = 1e-6


def pairwise_euclidean_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Numerically stable pairwise L2 distance matrix.

    embeddings: (N, D), assumed L2-normalized (as produced by
    FingerprintEmbeddingNet), so distances lie in [0, 2].
    """
    dot = embeddings @ embeddings.t()
    sq_norms = dot.diag().unsqueeze(0)
    dist_sq = sq_norms.t() + sq_norms - 2.0 * dot
    # Clamp to a real, non-tiny epsilon floor rather than 0.0. This keeps
    # near-duplicate pairs from landing exactly on a zero-gradient plateau
    # due to float32 cancellation noise -- see module docstring.
    dist_sq = dist_sq.clamp(min=_DIST_SQ_EPS)
    dist = torch.sqrt(dist_sq)
    return dist


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        embeddings: (N, D)
        labels:     (N,) integer subject ids (contiguous indices)
        returns: scalar loss, plus a dict of diagnostics
        """
        device = embeddings.device
        n = embeddings.size(0)
        dist = pairwise_euclidean_distances(embeddings)  # (N, N)

        labels = labels.to(device)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N) bool
        diff_label = ~same_label

        # Mask out self-comparisons (diagonal) from the positive mask.
        eye = torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = same_label & (~eye)
        negative_mask = diff_label

        if positive_mask.sum() == 0:
            raise ValueError(
                "No positive pairs found in this batch -- every subject "
                "appeared only once. Increase K in the PKSampler."
            )

        # Hardest positive per anchor: max distance among same-label pairs.
        # Where an anchor has no valid positive, dist is set to -inf so it
        # never wins the max (shouldn't happen given PKSampler, but safe).
        dist_pos = dist.masked_fill(~positive_mask, float("-inf"))
        hardest_positive, _ = dist_pos.max(dim=1)

        # Hardest negative per anchor: min distance among different-label pairs.
        dist_neg = dist.masked_fill(~negative_mask, float("inf"))
        hardest_negative, _ = dist_neg.min(dim=1)

        losses = torch.relu(hardest_positive - hardest_negative + self.margin)
        loss = losses.mean()

        with torch.no_grad():
            frac_active = (losses > 1e-6).float().mean().item()
            diagnostics = {
                "mean_hardest_pos_dist": hardest_positive.mean().item(),
                "mean_hardest_neg_dist": hardest_negative.mean().item(),
                "fraction_active_triplets": frac_active,
            }
        return loss, diagnostics
