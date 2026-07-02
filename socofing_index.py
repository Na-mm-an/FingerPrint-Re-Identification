"""
socofing_index.py

Framework-agnostic indexing of the SOCOFing dataset for subject identification
(open-set biometric re-identification).

SOCOFing filenames look like:
    "37__M_Left_index_finger.BMP"                  (Real)
    "37__M_Left_index_finger_CR.BMP"                (Altered, central rotation)
    "37__M_Left_index_finger_Obl.BMP"               (Altered, obliteration)
    "37__M_Left_index_finger_Zcut.BMP"              (Altered, z-cut)

We parse the subject id (first underscore-delimited token) along with hand
and finger, since the true "identity" for a fingerprint re-ID system is a
single FINGER, not a person -- see SocofingRecord.finger_uid below. Gender
is kept around as metadata for stratified sampling / diagnostics but is not
used as a label.

CHANGE (after observing that switching to finger-level identity was needed
to stop batch-hard triplet mining from being handed "positive" pairs that
were actually two different fingers of the same person -- e.g. thumb vs.
pinky -- which are structurally unrelated and were driving the network
toward embedding collapse regardless of optimizer/LR/weight-decay/numerics
fixes): added SocofingRecord.finger_uid, a (subject_id, hand, finger) key.
Callers that need per-finger identity (e.g. dataset.py's label maps for
triplet mining) should group/key on finger_uid. Callers that need
person-level identity (e.g. the train/val/test split below, so a person's
skin/texture characteristics can't leak across splits via a held-back
finger) should keep using subject_id, unchanged.
"""

from __future__ import annotations

import os
import glob
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SocofingRecord:
    path: str
    subject_id: int
    gender: str
    hand: str
    finger: str
    is_altered: bool
    alteration_level: Optional[str] = None  # "Easy" / "Medium" / "Hard" / None for Real

    @property
    def finger_uid(self) -> str:
        """Identity key for a single FINGER, not a person.

        subject_id alone groups all ~10 fingers of one person into a single
        "identity", which is wrong for triplet mining: different fingers of
        the same person have unrelated ridge patterns, so batch-hard mining
        would regularly pick a "hardest positive" pair that's e.g. one
        person's thumb vs. their pinky and push the network to make those
        look similar. Use (subject_id, hand, finger) as the true identity
        instead -- this is what "same finger, imaged multiple times / under
        different alteration levels" actually means in SOCOFing.
        """
        return f"{self.subject_id}_{self.hand}_{self.finger}"


def _parse_filename(path: str, is_altered: bool, alteration_level: Optional[str]) -> SocofingRecord:
    """Parse one SOCOFing filename into a structured record.

    Real:    1__M_Left_index_finger.BMP
    Altered: 1__M_Left_index_finger_CR.BMP / _Obl.BMP / _Zcut.BMP
    """
    fname = os.path.basename(path)
    stem = fname.rsplit(".", 1)[0]
    parts = stem.split("_")
    # parts example (Real):    ['1', '', 'M', 'Left', 'index', 'finger']
    # parts example (Altered): ['1', '', 'M', 'Left', 'index', 'finger', 'CR']
    subject_id = int(parts[0])
    gender = parts[2]
    hand = parts[3]
    finger = parts[4]
    return SocofingRecord(
        path=path,
        subject_id=subject_id,
        gender=gender,
        hand=hand,
        finger=finger,
        is_altered=is_altered,
        alteration_level=alteration_level,
    )


def index_socofing(root_dir: str, include_altered_levels: Optional[List[str]] = None) -> List[SocofingRecord]:
    """Walk a SOCOFing root directory and return a flat list of records.

    Parameters
    ----------
    root_dir : path to the folder that directly contains "Real" and "Altered".
    include_altered_levels : which Altered-* subfolders to include
        (default: all of Easy/Medium/Hard). Pass [] to use only Real images.
    """
    if include_altered_levels is None:
        include_altered_levels = ["Easy", "Medium", "Hard"]

    records: List[SocofingRecord] = []

    real_dir = os.path.join(root_dir, "Real")
    for path in sorted(glob.glob(os.path.join(real_dir, "*.BMP"))):
        records.append(_parse_filename(path, is_altered=False, alteration_level=None))

    for level in include_altered_levels:
        alt_dir = os.path.join(root_dir, "Altered", f"Altered-{level}")
        for path in sorted(glob.glob(os.path.join(alt_dir, "*.BMP"))):
            records.append(_parse_filename(path, is_altered=True, alteration_level=level))

    if not records:
        raise FileNotFoundError(
            f"No .BMP files found under {root_dir}. Expected a 'Real' folder and/or "
            f"'Altered/Altered-{{Easy,Medium,Hard}}' folders."
        )
    return records


def group_by_subject(records: List[SocofingRecord]) -> Dict[int, List[SocofingRecord]]:
    """Groups by PERSON (subject_id). Used only for the train/val/test
    split, where we want to hold out whole people, not individual fingers --
    see split_subjects_train_val_test. NOT used for triplet-mining labels;
    use group_by_finger for that.
    """
    grouped: Dict[int, List[SocofingRecord]] = defaultdict(list)
    for r in records:
        grouped[r.subject_id].append(r)
    return grouped


def group_by_finger(records: List[SocofingRecord]) -> Dict[str, List[SocofingRecord]]:
    """Groups by FINGER (finger_uid) -- the correct identity granularity
    for triplet-loss positive/negative mining. Different fingers of the
    same person are different identities.
    """
    grouped: Dict[str, List[SocofingRecord]] = defaultdict(list)
    for r in records:
        grouped[r.finger_uid].append(r)
    return grouped


def split_gallery_probe(records: List[SocofingRecord], seed: int = 42):
    """Enrollment/identification split.

    Gallery = the Real (unaltered) image for each subject/finger -> what a
    biometric system would have "on file".
    Probe    = the Altered images -> what a new scan looks like when the
    system tries to identify who it belongs to.

    This mirrors how SOCOFing was actually designed to be used, and avoids
    the leakage you'd get from a naive random image-level split (which could
    put the Real and Altered version of the *same* finger scan on both sides).
    """
    gallery = [r for r in records if not r.is_altered]
    probe = [r for r in records if r.is_altered]
    return gallery, probe


def split_subjects_train_val_test(
    subject_ids: List[int], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
):
    """Split at the SUBJECT (person) level -- not finger level, and not
    image level -- so that a given PERSON's fingers never appear in more
    than one split. This is deliberately more conservative than splitting
    by finger_uid: a person's fingers can share correlated skin/texture
    characteristics, so holding out only one of their ten fingers could
    still leak person-specific information into training. Held-out subjects
    therefore contribute zero fingers to training, exactly like a deployed
    system encountering a genuinely new person at enrollment time.
    """
    ids = sorted(subject_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])
    train_ids = set(ids[n_test + n_val:])
    return train_ids, val_ids, test_ids


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "SOCOFing"
    recs = index_socofing(root)
    grouped = group_by_subject(recs)
    finger_grouped = group_by_finger(recs)
    print(f"Total records: {len(recs)}")
    print(f"Subjects (people): {len(grouped)}")
    print(f"Fingers (identity classes for triplet mining): {len(finger_grouped)}")
    real_count = sum(1 for r in recs if not r.is_altered)
    alt_count = sum(1 for r in recs if r.is_altered)
    print(f"Real: {real_count}  Altered: {alt_count}")
    train_ids, val_ids, test_ids = split_subjects_train_val_test(list(grouped.keys()))
    print(f"Subject split -> train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")
