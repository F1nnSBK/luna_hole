"""
build_hf_dataset.py
===================
Builds the F1nnSBK/lunar-pits-dataset v2 HuggingFace dataset.

Schema per sample:
    sample_id        str    "Adams_B_1_M1149067652RC_aug3" | "..._original"
    source_tile_id   str    "Adams_B_1_M1149067652RC"  (traceable to original)
    lpa_id           int    LPA catalog ID (22–299), -1 for negatives
    crater_name      str    "Adams B" | "" for negatives
    pit_name         str    "Adams B 1" | "" for negatives
    host_type        str    "impact_melt" | "mare" | "highland" | ""
    latitude         float  Pit centroid latitude (LPA) | NaN for negatives
    longitude        float  Pit centroid longitude (LPA) | NaN for negatives
    funnel_max_m     float  Funnel major axis [m] | NaN
    funnel_min_m     float  Funnel minor axis [m] | NaN
    inner_max_m      float  Inner pit major axis [m] | NaN
    inner_min_m      float  Inner pit minor axis [m] | NaN
    depth_m          float  Pit depth [m] | NaN
    azimuth_deg      float  Azimuth of major axis | NaN
    nac_strip_id     str    "M1149067652RC"
    nac_camera       str    "RC" | "LC"
    label            int    1=pit, 0=negative
    split            str    "train" | "validation" | "test"
    is_augmented     bool
    augmentation_id  int    0=original, 1-8=augmentation index
    augmentation_ops str    human-readable ops applied, e.g. "hflip+rot90"
    nac_p2           float  2nd percentile of NAC strip (for denormalization)
    nac_p98          float  98th percentile of NAC strip
    image            PIL Image (224x224, uint8, 3-channel grayscale)
    npy_data         np.ndarray float32 (224x224), normalized [0,1]
    image_width      int    224
    image_height     int    224
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

# HuggingFace imports
try:
    from datasets import (
        Array2D, Dataset, DatasetDict, Features,
        Image as HFImage, Value,
    )
except ImportError:
    sys.exit("datasets not installed. Run: pip install datasets huggingface_hub pillow")

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).parent
DATA_DIR       = REPO_ROOT / "data" / "processed" / "dataset"
NAC_STATS_FILE = REPO_ROOT / "data" / "nac_stats.json"
LPA_CSV        = Path("/Users/finnhertsch/projects/luna/catalogs/lpa.csv")

HF_REPO_ID     = "F1nnSBK/lunar-pits-dataset"
TILE_SIZE      = 224
AUG_K          = 8          # augmentations per train pit tile
RANDOM_STATE   = 42
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
# test = 1.0 - TRAIN_FRAC - VAL_FRAC = 0.15


# ── Augmentation ops ───────────────────────────────────────────────────────────

def _augment_npy(arr: np.ndarray, aug_id: int) -> tuple[np.ndarray, str]:
    """Apply a deterministic augmentation to a float32 (H,W) numpy array.

    aug_id 0 = original (identity), 1-8 = augmentations.
    Returns (augmented_array, ops_description).
    """
    if aug_id == 0:
        return arr.copy(), "original"

    rng = np.random.default_rng(aug_id * 1337)
    a = arr.copy()
    ops = []

    # Rotation: 0/90/180/270 deterministically by aug_id
    rot_k = (aug_id - 1) % 4
    a = np.rot90(a, k=rot_k)
    if rot_k > 0:
        ops.append(f"rot{rot_k * 90}")

    # Horizontal flip
    if aug_id % 2 == 0:
        a = np.fliplr(a)
        ops.append("hflip")

    # Vertical flip
    if aug_id % 3 == 0:
        a = np.flipud(a)
        ops.append("vflip")

    # Gaussian noise (simulate NAC sensor noise)
    sigma = float(0.008 + rng.random() * 0.012)
    noise = rng.normal(0, sigma, a.shape).astype(np.float32)
    a = np.clip(a + noise, 0.0, 1.0)
    ops.append(f"gnoise_s{sigma:.3f}")

    # Brightness jitter (solar incidence angle variation)
    factor = float(0.85 + rng.random() * 0.30)
    a = np.clip(a * factor, 0.0, 1.0)
    ops.append(f"bright_{factor:.2f}")

    # Random erasing for aug_id >= 5 (simulate shadow occlusion)
    if aug_id >= 5:
        h, w = a.shape
        erase_h = int(h * float(0.05 + rng.random() * 0.10))
        erase_w = int(w * float(0.05 + rng.random() * 0.10))
        r0 = int(rng.integers(0, h - erase_h))
        c0 = int(rng.integers(0, w - erase_w))
        a[r0:r0 + erase_h, c0:c0 + erase_w] = float(rng.random() * 0.15)
        ops.append("shadow_erase")

    return a.astype(np.float32), "+".join(ops) if ops else "original"


# ── LPA catalog ────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_lpa(csv_path: Path) -> dict[str, dict]:
    """Load LPA catalog keyed by NFC-normalised pit name."""
    lpa: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = _norm(row["name"])
            lpa[key] = {
                "lpa_id":       int(row["id"]),
                "crater_name":  row["host"],
                "host_type":    row["host_type"],
                "latitude":     float(row["latitude"]) if row["latitude"] else float("nan"),
                "longitude":    float(row["longitude"]) if row["longitude"] else float("nan"),
                "funnel_max_m": float(row["funnel_max_m"]) if row["funnel_max_m"] else float("nan"),
                "funnel_min_m": float(row["funnel_min_m"]) if row["funnel_min_m"] else float("nan"),
                "inner_max_m":  float(row["inner_max_m"]) if row["inner_max_m"] else float("nan"),
                "inner_min_m":  float(row["inner_min_m"]) if row["inner_min_m"] else float("nan"),
                "depth_m":      float(row["depth_m"]) if row["depth_m"] else float("nan"),
                "azimuth_deg":  float(row["azimuth_deg"]) if row["azimuth_deg"] else float("nan"),
            }
    return lpa


def nac_strip_from_stem(stem: str) -> str:
    """'Adams_B_1_M1149067652RC' -> 'M1149067652RC'"""
    return stem.rsplit("_", 1)[-1]


def pit_name_from_stem(stem: str) -> str:
    """'Adams_B_1_M1149067652RC' -> 'Adams B 1'"""
    return stem.rsplit("_", 1)[0].replace("_", " ")


# ── Tile I/O ───────────────────────────────────────────────────────────────────

def load_npy_normalised(
    npy_path: Path, nac_stats: dict
) -> tuple[np.ndarray, float, float]:
    """Load and normalise using strip-level P2/P98. Returns (arr, p2, p98)."""
    arr = np.load(npy_path).astype(np.float32)
    nac_id = nac_strip_from_stem(npy_path.stem)
    if nac_id in nac_stats:
        p2  = float(nac_stats[nac_id]["min"])
        p98 = float(nac_stats[nac_id]["max"])
    else:
        valid = arr[arr > -32752]
        p2  = float(np.percentile(valid, 2)) if valid.size else 0.0
        p98 = float(np.percentile(valid, 98)) if valid.size else 1.0
    rng_val = p98 - p2
    arr = np.clip(arr, p2, p98)
    arr = (arr - p2) / rng_val if rng_val > 0 else np.zeros_like(arr)
    return arr.astype(np.float32), p2, p98


def resize_npy(arr: np.ndarray, size: int = TILE_SIZE) -> np.ndarray:
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(arr_u8, mode="L").resize((size, size), Image.BICUBIC)
    return np.array(pil, dtype=np.float32) / 255.0


def npy_to_pil(arr: np.ndarray) -> Image.Image:
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="L").convert("RGB")


# ── 3-way split ────────────────────────────────────────────────────────────────

def three_way_group_split(
    pit_files: list[Path],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    random_state: int = RANDOM_STATE,
) -> tuple[set[str], set[str], set[str]]:
    """Split unique NAC strip IDs into train/val/test — no strip crosses splits."""
    strip_ids = list({nac_strip_from_stem(p.stem) for p in pit_files})
    rng = random.Random(random_state)
    rng.shuffle(strip_ids)
    n = len(strip_ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return (
        set(strip_ids[:n_train]),
        set(strip_ids[n_train:n_train + n_val]),
        set(strip_ids[n_train + n_val:]),
    )


def assign_split(nac_id: str, train: set, val: set) -> str:
    if nac_id in train:
        return "train"
    elif nac_id in val:
        return "validation"
    return "test"


# ── Sample building ────────────────────────────────────────────────────────────

def _empty_geo() -> dict:
    return {
        "lpa_id": -1, "pit_name": "", "crater_name": "", "host_type": "",
        "latitude": float("nan"), "longitude": float("nan"),
        "funnel_max_m": float("nan"), "funnel_min_m": float("nan"),
        "inner_max_m": float("nan"), "inner_min_m": float("nan"),
        "depth_m": float("nan"), "azimuth_deg": float("nan"),
    }


def make_sample(
    stem: str,
    arr_resized: np.ndarray,
    label: int,
    split_name: str,
    nac_stats: dict,
    lpa: dict,
    npy_path: Path,
    aug_id: int = 0,
    aug_ops: str = "original",
) -> dict:
    nac_id = nac_strip_from_stem(stem)
    _, p2, p98 = load_npy_normalised(npy_path, nac_stats)

    if label == 1:
        pname = pit_name_from_stem(stem)
        geo = lpa.get(_norm(pname), {})
        geo_fields = {
            "lpa_id":       geo.get("lpa_id", -1),
            "pit_name":     pname,
            "crater_name":  geo.get("crater_name", ""),
            "host_type":    geo.get("host_type", ""),
            "latitude":     geo.get("latitude", float("nan")),
            "longitude":    geo.get("longitude", float("nan")),
            "funnel_max_m": geo.get("funnel_max_m", float("nan")),
            "funnel_min_m": geo.get("funnel_min_m", float("nan")),
            "inner_max_m":  geo.get("inner_max_m", float("nan")),
            "inner_min_m":  geo.get("inner_min_m", float("nan")),
            "depth_m":      geo.get("depth_m", float("nan")),
            "azimuth_deg":  geo.get("azimuth_deg", float("nan")),
        }
    else:
        geo_fields = _empty_geo()

    suffix = f"_aug{aug_id}" if aug_id > 0 else "_original"
    return {
        "sample_id":        stem + suffix,
        "source_tile_id":   stem,
        **geo_fields,
        "nac_strip_id":     nac_id,
        "nac_camera":       nac_id[-2:],
        "label":            label,
        "split":            split_name,
        "is_augmented":     aug_id > 0,
        "augmentation_id":  aug_id,
        "augmentation_ops": aug_ops,
        "nac_p2":           p2,
        "nac_p98":          p98,
        "image":            npy_to_pil(arr_resized),
        "npy_data":         arr_resized,
        "image_width":      TILE_SIZE,
        "image_height":     TILE_SIZE,
    }


# ── HF Features ────────────────────────────────────────────────────────────────

def build_features() -> Features:
    return Features({
        "sample_id":        Value("string"),
        "source_tile_id":   Value("string"),
        "lpa_id":           Value("int32"),
        "pit_name":         Value("string"),
        "crater_name":      Value("string"),
        "host_type":        Value("string"),
        "latitude":         Value("float32"),
        "longitude":        Value("float32"),
        "funnel_max_m":     Value("float32"),
        "funnel_min_m":     Value("float32"),
        "inner_max_m":      Value("float32"),
        "inner_min_m":      Value("float32"),
        "depth_m":          Value("float32"),
        "azimuth_deg":      Value("float32"),
        "nac_strip_id":     Value("string"),
        "nac_camera":       Value("string"),
        "label":            Value("int8"),
        "split":            Value("string"),
        "is_augmented":     Value("bool"),
        "augmentation_id":  Value("int8"),
        "augmentation_ops": Value("string"),
        "nac_p2":           Value("float32"),
        "nac_p98":          Value("float32"),
        "image":            HFImage(),
        "npy_data":         Array2D(shape=(TILE_SIZE, TILE_SIZE), dtype="float32"),
        "image_width":      Value("int32"),
        "image_height":     Value("int32"),
    })


# ── Main ───────────────────────────────────────────────────────────────────────

def main(push: bool = False, dry_run: bool = False) -> None:
    print("=" * 60)
    print("lunar-pits-dataset v2 builder")
    print("=" * 60)

    # 1. Load support files
    print("\n[1/6] Loading NAC stats and LPA catalog...")
    with open(NAC_STATS_FILE) as f:
        nac_stats: dict = json.load(f)
    lpa = load_lpa(LPA_CSV)
    print(f"      NAC stats : {len(nac_stats)} strips")
    print(f"      LPA catalog: {len(lpa)} pits")

    # 2. Collect all pit .npy paths
    print("\n[2/6] Collecting pit tiles...")
    all_pit_npy: list[Path] = []
    for sub in ["train", "test"]:
        d = DATA_DIR / sub / "pits"
        if d.exists():
            all_pit_npy.extend(sorted(d.glob("*.npy")))
    print(f"      Found {len(all_pit_npy)} pit tiles")

    # 3. 3-way strip split
    print("\n[3/6] 3-way GroupSplit by NAC strip ID...")
    train_strips, val_strips, test_strips = three_way_group_split(all_pit_npy)
    print(f"      Train: {len(train_strips)} strips | "
          f"Val: {len(val_strips)} | Test: {len(test_strips)}")

    manifest = {
        "random_state": RANDOM_STATE, "train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC,
        "aug_k": AUG_K, "tile_size": TILE_SIZE,
        "train_strips": sorted(train_strips),
        "val_strips":   sorted(val_strips),
        "test_strips":  sorted(test_strips),
    }
    manifest_path = REPO_ROOT / "data" / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"      Manifest -> {manifest_path}")

    if dry_run:
        print("\n[DRY RUN] Schema and split validated. Done.")
        return

    # 4. Load and sort all samples
    print("\n[4/6] Loading and normalising all tiles...")
    train_records, val_records, test_records = [], [], []

    # Pits
    for npy_path in all_pit_npy:
        stem = npy_path.stem
        nac_id = nac_strip_from_stem(stem)
        split_name = assign_split(nac_id, train_strips, val_strips)
        arr, p2, p98 = load_npy_normalised(npy_path, nac_stats)
        arr_r = resize_npy(arr)
        rec = make_sample(stem, arr_r, 1, split_name, nac_stats, lpa, npy_path)
        # Override p2/p98 with already-computed values
        rec["nac_p2"] = p2; rec["nac_p98"] = p98
        if split_name == "train":
            train_records.append((rec, arr_r, npy_path, stem))
        elif split_name == "validation":
            val_records.append(rec)
        else:
            test_records.append(rec)

    # Negatives
    for sub in ["train", "test"]:
        neg_dir = DATA_DIR / sub / "negatives"
        if not neg_dir.exists():
            continue
        for npy_path in sorted(neg_dir.glob("*.npy")):
            stem = npy_path.stem
            nac_id = nac_strip_from_stem(stem)
            split_name = assign_split(nac_id, train_strips, val_strips)
            arr, p2, p98 = load_npy_normalised(npy_path, nac_stats)
            arr_r = resize_npy(arr)
            rec = make_sample(stem, arr_r, 0, split_name, nac_stats, lpa, npy_path)
            rec["nac_p2"] = p2; rec["nac_p98"] = p98
            if split_name == "train":
                train_records.append((rec, arr_r, npy_path, stem))
            elif split_name == "validation":
                val_records.append(rec)
            else:
                test_records.append(rec)

    # Extract plain records from train (tuple form was for augmentation)
    train_plain = [t[0] for t in train_records]
    pit_train_tuples = [(t[1], t[2], t[3]) for t in train_records if t[0]["label"] == 1]

    print(f"      Train:  {sum(r['label']==1 for r in train_plain)} pits, "
          f"{sum(r['label']==0 for r in train_plain)} negatives")
    print(f"      Val:    {sum(r['label']==1 for r in val_records)} pits, "
          f"{sum(r['label']==0 for r in val_records)} negatives")
    print(f"      Test:   {sum(r['label']==1 for r in test_records)} pits, "
          f"{sum(r['label']==0 for r in test_records)} negatives")

    # 5. Augment train pits
    print(f"\n[5/6] Augmenting {len(pit_train_tuples)} train pit tiles x{AUG_K}...")
    aug_records = []
    for arr_r, npy_path, stem in pit_train_tuples:
        for aug_id in range(1, AUG_K + 1):
            arr_aug, ops = _augment_npy(arr_r, aug_id)
            rec = make_sample(stem, arr_aug, 1, "train", nac_stats, lpa, npy_path,
                              aug_id=aug_id, aug_ops=ops)
            aug_records.append(rec)

    all_train = train_plain + aug_records
    print(f"      Total train after aug: {len(all_train)} samples "
          f"({sum(r['label']==1 for r in all_train)} pits, "
          f"{sum(r['label']==0 for r in all_train)} negatives)")

    # 6. Build DatasetDict
    print("\n[6/6] Building HuggingFace DatasetDict...")
    features = build_features()

    def to_ds(records: list[dict]) -> Dataset:
        return Dataset.from_list(records, features=features)

    ds = DatasetDict({
        "train":      to_ds(all_train),
        "validation": to_ds(val_records),
        "test":       to_ds(test_records),
    })
    print(ds)

    local_path = REPO_ROOT / "data" / "hf_dataset"
    print(f"\nSaving locally to {local_path}...")
    ds.save_to_disk(str(local_path))
    print("Saved locally.")

    if push:
        print(f"\nPushing to {HF_REPO_ID}...")
        ds.push_to_hub(
            HF_REPO_ID,
            private=False,
            commit_message="v2: 3-way split, georeferencing (LPA), k=8 aug, NPY in Parquet",
        )
        print(f"Done! https://huggingface.co/datasets/{HF_REPO_ID}")
    else:
        print("\nRe-run with --push to upload to HuggingFace Hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build lunar-pits-dataset v2")
    parser.add_argument("--push",    action="store_true", help="Push to HuggingFace Hub after build")
    parser.add_argument("--dry-run", action="store_true", help="Validate splits only, skip dataset creation")
    args = parser.parse_args()
    main(push=args.push, dry_run=args.dry_run)
