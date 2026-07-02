"""
dataset.py

PyTorch Dataset + P-K batch sampler for SOCOFing subject identification.

Why a P-K sampler: batch-hard triplet loss needs, within every batch, several
images from the SAME subject (to mine a hard positive) and images from
several DIFFERENT subjects (to mine a hard negative). A batch built by plain
random shuffling will often contain zero or one image per subject, which
gives the loss nothing to mine. A P-K sampler guarantees P distinct subjects
x K images each per batch (batch size = P*K).

CHANGE (root-cause fix for embedding collapse): label_to_idx now keys on
SocofingRecord.finger_uid (subject_id + hand + finger) instead of raw
subject_id. subject_id alone groups all ~10 fingers of one person into a
single triplet-mining "identity", so batch-hard mining was regularly
selecting a "hardest positive" pair that was actually two different,
structurally unrelated fingers of the same person (e.g. thumb vs. pinky)
and pushing the network to make them look similar. That is a direct,
data-level cause of collapse, independent of LR/weight-decay/optimizer
numerics -- see train.py and socofing_index.py for the rest of the
diagnosis. Person-level identity is still used correctly elsewhere (the
train/val/test split in train.py/socofing_index.py), since holding out
whole people rather than individual fingers is still the right way to
prevent leakage across splits.
"""

from __future__ import annotations

import random
from typing import List, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

from socofing_index import SocofingRecord, group_by_subject


class SocofingReIDDataset(Dataset):
    """Returns (image_tensor, finger_label_index) pairs.

    finger_label_index is a 0..N-1 contiguous index (not the raw SOCOFing
    subject id, and not a raw finger_uid string), built from label_to_idx
    so it can be used as a tensor. It identifies a single FINGER, since
    that's the correct identity granularity for triplet-loss positive
    pairs -- see module docstring.
    """

    def __init__(
        self,
        records: List[SocofingRecord],
        label_to_idx: Dict[str, int],
        image_size: int = 96,
        train: bool = True,
    ):
        self.records = records
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.train = train

        # Precompute, for the sampler, which dataset indices belong to which
        # contiguous label.
        self.label_to_indices: Dict[int, List[int]] = {}
        for i, r in enumerate(self.records):
            lbl = self.label_to_idx[r.finger_uid]
            self.label_to_indices.setdefault(lbl, []).append(i)

    def __len__(self):
        return len(self.records)

    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        # Light, dependency-free augmentation (no imgaug/torchvision needed):
        # random horizontal jitter via small translation + random brightness.
        # Kept deliberately mild -- fingerprint orientation matters, so we
        # avoid rotation/flip augmentations that would distort ridge geometry
        # in ways that don't occur in the real capture process.
        if random.random() < 0.5:
            shift = random.randint(-4, 4)
            arr = np.roll(arr, shift, axis=1)
        if random.random() < 0.5:
            arr = np.clip(arr * random.uniform(0.85, 1.15), 0.0, 1.0)
        return arr

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        arr = self._load_image(rec.path)
        if self.train:
            arr = self._augment(arr)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()  # (1, H, W)
        label = self.label_to_idx[rec.finger_uid]
        return tensor, label


class PKSampler(Sampler):
    """Yields batches of P*K indices: P distinct labels, K samples each.

    If a label has fewer than K available images, its images are sampled
    WITH replacement to still fill K slots (SOCOFing subjects can have very
    few images for a given finger, especially once altered images are
    filtered by difficulty level).
    """

    def __init__(self, dataset: SocofingReIDDataset, p: int, k: int, iterations: int):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.iterations = iterations
        self.labels = list(dataset.label_to_indices.keys())
        if len(self.labels) < p:
            raise ValueError(
                f"P={p} distinct subjects requested per batch but only "
                f"{len(self.labels)} subjects are available in this split."
            )

    def __len__(self):
        return self.iterations

    def __iter__(self):
        for _ in range(self.iterations):
            chosen_labels = random.sample(self.labels, self.p)
            batch = []
            for lbl in chosen_labels:
                pool = self.dataset.label_to_indices[lbl]
                if len(pool) >= self.k:
                    batch.extend(random.sample(pool, self.k))
                else:
                    batch.extend(random.choices(pool, k=self.k))
            random.shuffle(batch)
            yield batch


def build_label_map(all_records: List[SocofingRecord]) -> Dict[str, int]:
    """Maps finger_uid (subject_id + hand + finger) -> contiguous 0..N-1
    index. This is the identity granularity used for triplet-loss
    positive/negative mining -- see module docstring for why this is
    finger_uid rather than raw subject_id.
    """
    finger_uids = sorted({r.finger_uid for r in all_records})
    return {fid: i for i, fid in enumerate(finger_uids)}
