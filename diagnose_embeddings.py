"""
diagnose_embeddings.py

Confirms (or rules out) embedding collapse: loads a checkpoint, embeds a
handful of real images from several different subjects, and reports the
spread of pairwise distances. If collapsed, ALL distances (same-subject and
different-subject alike) will be tiny and nearly identical.

Usage:
    python diagnose_embeddings.py --data_root /path/to/SOCOFing --checkpoint ./checkpoints/best_model.pt
"""

import argparse
import numpy as np
import torch

from socofing_index import index_socofing, group_by_subject
from dataset import SocofingReIDDataset, build_label_map
from model import FingerprintEmbeddingNet

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--n_subjects", type=int, default=10)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.checkpoint, map_location=device)
model = FingerprintEmbeddingNet(embedding_dim=ckpt["embedding_dim"]).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

records = index_socofing(args.data_root, include_altered_levels=[])  # Real only, fast
grouped = group_by_subject(records)
subject_ids = list(grouped.keys())[: args.n_subjects]
sample_records = [grouped[sid][0] for sid in subject_ids]  # one image per subject

label_map = build_label_map(sample_records)
ds = SocofingReIDDataset(sample_records, label_map, image_size=ckpt["image_size"], train=False)

with torch.no_grad():
    imgs = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    embeddings = model(imgs).cpu().numpy()

dist = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)
off_diag = dist[~np.eye(dist.shape[0], dtype=bool)]

print(f"Embedded {len(sample_records)} images from {len(sample_records)} different subjects")
print(f"Pairwise distance stats (all different subjects, since 1 image each):")
print(f"  mean: {off_diag.mean():.4f}")
print(f"  std:  {off_diag.std():.4f}")
print(f"  min:  {off_diag.min():.4f}")
print(f"  max:  {off_diag.max():.4f}")

if off_diag.std() < 0.02 and off_diag.mean() < 0.1:
    print("\n>>> COLLAPSE CONFIRMED: distances are tiny and nearly uniform.")
    print(">>> The model maps all fingerprints to ~the same embedding point.")
else:
    print("\n>>> Distances show real spread -- collapse is unlikely; loss curve issue may be something else.")
