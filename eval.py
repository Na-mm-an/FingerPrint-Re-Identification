"""
eval.py

Open-set evaluation for the fingerprint embedding model, on subjects that
were NEVER seen during training (test split from split_subjects_train_val_test).

This is the number that actually matters -- not train/val triplet loss.
Triplet loss just measures whether *sampled* triplets satisfy a margin,
which gets trivially easy once classes separate even a little. These two
metrics measure the thing a deployed biometric system actually needs:

  1. IDENTIFICATION (closed-set, 1:N matching):
     Gallery = Real (unaltered) image per finger -- what the system has
     "on file" from enrollment.
     Probe   = Altered images -- what a new scan looks like.
     For each probe, is the nearest gallery embedding the SAME finger?
     Reported as rank-1 and rank-5 accuracy.

  2. VERIFICATION (1:1 matching, open-set):
     For every probe, compare against every gallery entry. A pair is
     "genuine" if same finger_uid, "impostor" otherwise. Sweep a distance
     threshold and report Equal Error Rate (EER): the threshold where
     False Accept Rate == False Reject Rate. Lower EER = better.

Why this specifically checks for the "shortcut" concern raised during
training: if the model were mostly keying off alteration-type artifacts
(CR/Obl/Zcut visual signatures) rather than genuine ridge structure, it
would still do reasonably at grouping altered images of DIFFERENT fingers
that share an alteration type, which would show up as elevated impostor
scores / a worse EER and lower rank-1 accuracy than the near-zero training
loss would suggest. Real ridge-structure learning should generalize fine
across alteration types since gallery is always Real (unaltered).

Usage:
    python eval.py --data_root /path/to/SOCOFing --checkpoint ./checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from socofing_index import index_socofing, group_by_subject, split_subjects_train_val_test, split_gallery_probe
from dataset import SocofingReIDDataset, build_label_map
from model import FingerprintEmbeddingNet


def build_test_records(data_root: str, include_altered_levels, seed: int, val_frac=0.15, test_frac=0.15):
    """Mirrors train.py's build_splits, but we only need the test subjects
    here. Uses the same default seed (42) as train.py so this reproduces
    the exact same held-out subject set the model never saw.
    """
    records = index_socofing(data_root, include_altered_levels=include_altered_levels)
    grouped = group_by_subject(records)
    _, _, test_ids = split_subjects_train_val_test(
        list(grouped.keys()), val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    return [r for r in records if r.subject_id in test_ids]


@torch.no_grad()
def embed_records(model, records, label_to_idx, image_size, device, batch_size=128, num_workers=4):
    """Embeds a list of records, returns (embeddings [N,D] np.ndarray,
    finger_uids [N] list). Uses train=False so no augmentation is applied --
    we want a clean, deterministic embedding for evaluation.
    """
    ds = SocofingReIDDataset(records, label_to_idx, image_size=image_size, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    all_embeddings = []
    idx_to_finger_uid = {v: k for k, v in label_to_idx.items()}
    all_finger_uids = []

    for images, labels in loader:
        images = images.to(device)
        emb = model(images).cpu().numpy()
        all_embeddings.append(emb)
        all_finger_uids.extend(idx_to_finger_uid[int(l)] for l in labels)

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, all_finger_uids


def identification_accuracy(probe_emb, probe_ids, gallery_emb, gallery_ids, ranks=(1, 5)):
    """For each probe, rank all gallery entries by ascending L2 distance
    and check whether the correct finger_uid appears in the top-k.
    """
    # (N_probe, N_gallery) distance matrix, computed in chunks to bound
    # memory for large probe sets.
    gallery_ids = np.array(gallery_ids)
    correct_at = {r: 0 for r in ranks}
    max_rank = max(ranks)
    n_probe = probe_emb.shape[0]

    chunk = 256
    for start in range(0, n_probe, chunk):
        end = min(start + chunk, n_probe)
        batch_emb = probe_emb[start:end]  # (B, D)
        # Euclidean distance via ||a-b||^2 = ||a||^2+||b||^2-2ab, embeddings
        # are L2-normalized so this is well-conditioned here (not near 0
        # for the vast majority of pairs -- only true genuine matches will
        # be close, which is exactly what we want to detect).
        dists = np.linalg.norm(batch_emb[:, None, :] - gallery_emb[None, :, :], axis=2)  # (B, N_gallery)
        order = np.argsort(dists, axis=1)
        for i in range(end - start):
            probe_id = probe_ids[start + i]
            top_ids = gallery_ids[order[i, :max_rank]]
            for r in ranks:
                if probe_id in top_ids[:r]:
                    correct_at[r] += 1

    return {r: correct_at[r] / n_probe for r in ranks}


def verification_eer(probe_emb, probe_ids, gallery_emb, gallery_ids, max_impostor_pairs=200_000, seed=42):
    """Computes Equal Error Rate over genuine vs. impostor probe-gallery
    pairs. Genuine = same finger_uid. Impostor pairs are subsampled if the
    full cross product is too large, since N_probe x N_gallery can be huge.
    """
    rng = np.random.default_rng(seed)
    gallery_ids_arr = np.array(gallery_ids)

    genuine_scores = []
    impostor_scores = []

    # Build a lookup from finger_uid -> gallery index for fast genuine-pair
    # distance lookup (gallery has exactly one Real image per finger).
    gallery_index = {fid: i for i, fid in enumerate(gallery_ids)}

    for i, pid in enumerate(probe_ids):
        if pid in gallery_index:
            g_idx = gallery_index[pid]
            d = np.linalg.norm(probe_emb[i] - gallery_emb[g_idx])
            genuine_scores.append(d)

    n_probe = probe_emb.shape[0]
    n_gallery = gallery_emb.shape[0]
    total_possible_impostor = n_probe * n_gallery - len(genuine_scores)
    n_impostor_samples = min(max_impostor_pairs, total_possible_impostor)

    sampled = 0
    while sampled < n_impostor_samples:
        i = rng.integers(0, n_probe)
        j = rng.integers(0, n_gallery)
        if gallery_ids_arr[j] == probe_ids[i]:
            continue
        d = np.linalg.norm(probe_emb[i] - gallery_emb[j])
        impostor_scores.append(d)
        sampled += 1

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Sweep thresholds across the observed score range; FAR = fraction of
    # impostor pairs with distance <= threshold (falsely accepted), FRR =
    # fraction of genuine pairs with distance > threshold (falsely
    # rejected). EER is where the two curves cross.
    thresholds = np.linspace(0.0, 2.0, 2000)  # embeddings are L2-normalized -> dist in [0,2]
    far = np.array([(impostor_scores <= t).mean() for t in thresholds])
    frr = np.array([(genuine_scores > t).mean() for t in thresholds])
    diff = np.abs(far - frr)
    eer_idx = int(np.argmin(diff))
    eer = (far[eer_idx] + frr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
        "mean_genuine_dist": genuine_scores.mean(),
        "mean_impostor_dist": impostor_scores.mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--include_altered_levels", type=str, default="Easy,Medium,Hard")
    parser.add_argument("--seed", type=int, default=42, help="Must match the seed used in train.py")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_impostor_pairs", type=int, default=200_000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    levels = [s for s in args.include_altered_levels.split(",") if s]
    test_records = build_test_records(args.data_root, levels, seed=args.seed)
    gallery_records, probe_records = split_gallery_probe(test_records)
    print(f"Test subjects -> gallery (Real): {len(gallery_records)}  probe (Altered): {len(probe_records)}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = FingerprintEmbeddingNet(embedding_dim=ckpt["embedding_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    image_size = ckpt["image_size"]
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # Label map only needs to be internally consistent for this eval run;
    # it does NOT need to match the one used during training.
    label_to_idx = build_label_map(test_records)

    print("Embedding gallery...")
    gallery_emb, gallery_ids = embed_records(
        model, gallery_records, label_to_idx, image_size, device, args.batch_size, args.num_workers
    )
    print("Embedding probes...")
    probe_emb, probe_ids = embed_records(
        model, probe_records, label_to_idx, image_size, device, args.batch_size, args.num_workers
    )

    print("\n--- Identification (probe -> gallery, rank-k) ---")
    id_acc = identification_accuracy(probe_emb, probe_ids, gallery_emb, gallery_ids, ranks=(1, 5))
    for r, acc in id_acc.items():
        print(f"Rank-{r} accuracy: {acc:.4%}")

    print("\n--- Verification (EER) ---")
    ver = verification_eer(probe_emb, probe_ids, gallery_emb, gallery_ids, max_impostor_pairs=args.max_impostor_pairs, seed=args.seed)
    print(f"EER: {ver['eer']:.4%} @ threshold {ver['eer_threshold']:.4f}")
    print(f"Genuine pairs: {ver['n_genuine_pairs']}  mean dist: {ver['mean_genuine_dist']:.4f}")
    print(f"Impostor pairs: {ver['n_impostor_pairs']}  mean dist: {ver['mean_impostor_dist']:.4f}")


if __name__ == "__main__":
    main()
