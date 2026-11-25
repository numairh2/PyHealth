"""Prepare PMData user-day splits for the baseline and label-efficiency runs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy.special import ndtri

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyhealth.datasets import PMDataDataset, SampleDataset, create_sample_dataset
from pyhealth.tasks import PMDataPretraining, StressPredictionPMData
from pyhealth.tasks.clip_pretraining import DAILY_FEATURES, HOURLY_FEATURES, STATIC_FEATURES

FOLDS = tuple(range(5))
SEEDS = (0, 1)
LABEL_PCTS = (100, 50, 25, 10)
SPLIT_SEED = 2026
FEATURE_KEYS = (*HOURLY_FEATURES, "wellness", "static", "texts")


@dataclass
class UserDays:
    features: dict[tuple[str, str], dict]
    labels: dict[tuple[str, str], int]


def collect_days(dataset: PMDataDataset) -> UserDays:
    """Keep all Fitbit days for pretraining, and obtain observed stress from the team task."""
    pretrain_task, stress_task = PMDataPretraining(), StressPredictionPMData()
    features, labels = {}, {}
    for pid in sorted(dataset.unique_patient_ids):
        patient = dataset.get_patient(pid)
        for sample in pretrain_task(patient):
            features[pid, sample["record_id"]] = sample
        for sample in stress_task(patient):
            labels[pid, sample["record_id"]] = int(sample["stress"])
    return UserDays(features, labels)


def make_manifest(days: UserDays) -> pd.DataFrame:
    """Rotate each participant's days over five folds; users with <5 labels stay in train."""
    keys = sorted(days.features)
    patients = np.asarray([pid for pid, _ in keys])
    labels = np.asarray([days.labels.get(key, -1) for key in keys], dtype=int)
    outer = np.full(len(keys), -1, dtype=int)
    rng = np.random.default_rng(SPLIT_SEED)
    for pid in sorted(set(patients)):
        person = np.flatnonzero(patients == pid)
        if np.count_nonzero(labels[person] >= 0) < len(FOLDS):
            continue
        order = rng.permutation(FOLDS)
        offset = 0
        for label in (0, 1, -1):
            indices = rng.permutation(person[labels[person] == label])
            slots = (np.arange(len(indices)) + offset) % len(FOLDS)
            outer[indices] = order[slots]
            offset += len(indices)

    frames = []
    for fold in FOLDS:
        split = np.full(len(keys), "train", dtype=object)
        split[(outer == fold) & (labels >= 0)] = "test"
        split[(outer == fold) & (labels == -1)] = "heldout_unlabeled"
        val_rng = np.random.default_rng(SPLIT_SEED + fold + 1)
        for label in (0, 1):
            candidates = np.flatnonzero(
                (outer >= 0) & (outer != fold) & (labels == label)
            )
            if len(candidates) < 2 or not np.any((outer == fold) & (labels == label)):
                raise ValueError(f"Fold {fold} needs both classes in train/val/test.")
            count = min(len(candidates) - 1, max(1, int(len(candidates) * 0.10 + 0.5)))
            split[val_rng.choice(candidates, count, replace=False)] = "val"

        frame = pd.DataFrame({
            "fold": fold,
            "patient_id": patients,
            "record_id": [date for _, date in keys],
            "stress": pd.array([days.labels.get(key) for key in keys], dtype="Int8"),
            "outer_fold": outer,  # -1 denotes training-only participants
            "split": split,
        })
        for seed in SEEDS:
            subset_rng = np.random.default_rng(10000 + 100 * fold + seed)
            min_pct = np.zeros(len(keys), dtype=int)
            for label in (0, 1):
                train = np.flatnonzero((split == "train") & (labels == label))
                ranked = subset_rng.permutation(train)
                for pct in LABEL_PCTS:
                    count = max(1, (len(ranked) * pct + 50) // 100)
                    min_pct[ranked[:count]] = pct
            frame[f"min_pct_seed_{seed}"] = min_pct
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def normalized_features(days: UserDays, keys: list, train_mask: np.ndarray) -> dict:
    """Fit empirical normal scores on training observations. Leave NaN for the model."""
    arrays = {
        name: np.stack([days.features[key][name] for key in keys])
        for name in (*HOURLY_FEATURES, "wellness", "static")
    }
    columns = [arrays[name] for name in HOURLY_FEATURES]
    columns.extend(arrays["wellness"][:, i] for i in range(len(DAILY_FEATURES)))
    columns.extend(arrays["static"][:, i] for i, name in enumerate(STATIC_FEATURES) if name != "gender")
    for column in columns:
        values = column[train_mask]
        observed = values[np.isfinite(values)]
        present = np.isfinite(column)
        if not observed.size:
            column.fill(np.nan)
        elif observed.min() == observed.max():
            column[present] = 0
        else:
            ranks = np.linspace(0, 1, min(1000, observed.size))
            knots = np.quantile(observed, ranks)
            upper = np.interp(column[present], knots, ranks)
            lower = -np.interp(-column[present], -knots[::-1], -ranks[::-1])
            column[present] = ndtri(np.clip((upper + lower) / 2, 1e-7, 1 - 1e-7))
    return arrays


@dataclass
class FoldData:
    pretrain: SampleDataset
    train: SampleDataset
    val: SampleDataset
    test: SampleDataset
    train_rows: pd.DataFrame

    def label_indices(self, seed: int, pct: int) -> np.ndarray:
        if pct == 100:
            return np.arange(len(self.train))
        return np.flatnonzero(self.train_rows[f"min_pct_seed_{seed}"].between(1, pct))


def build_fold(days: UserDays, rows: pd.DataFrame) -> FoldData:
    rows = rows.sort_values(["patient_id", "record_id"]).reset_index(drop=True)
    keys = list(zip(rows.patient_id, rows.record_id))
    is_train = rows.split.eq("train").to_numpy()
    arrays = normalized_features(days, keys, is_train)
    samples = [
        {**days.features[key], **{name: column[i] for name, column in arrays.items()}}
        for i, key in enumerate(keys)
    ]
    pretrain = create_sample_dataset(
        [samples[i] for i in np.flatnonzero(is_train)],
        PMDataPretraining.input_schema, PMDataPretraining.output_schema, in_memory=True,
    )
    labeled = {}
    for split in ("train", "val", "test"):
        indices = np.flatnonzero(rows.split.eq(split) & rows.stress.notna())
        labeled[split] = create_sample_dataset(
            [{**samples[i], "stress": int(rows.at[i, "stress"])} for i in indices],
            StressPredictionPMData.input_schema, StressPredictionPMData.output_schema,
            input_processors=pretrain.input_processors,
            output_processors=labeled["train"].output_processors if split != "train" else None,
            in_memory=True,
        )
    train_rows = rows.loc[is_train & rows.stress.notna()]
    return FoldData(pretrain, labeled["train"], labeled["val"], labeled["test"], train_rows)


def prepare(root: Path, out: Path):
    """Load the team's dataset and write the user-day assignments shared by both methods."""
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / "prepare.lock")):
        dataset = PMDataDataset(root=str(root), cache_dir=str(out / "cache"))
        days = collect_days(dataset)
        manifest = make_manifest(days)
        path = out / "manifest.csv"
        content = manifest.to_csv(index=False, lineterminator="\n")
        if path.is_file():
            if path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Different user-day assignments already exist in {out}.")
        else:
            pending = path.with_suffix(".partial")
            pending.write_bytes(content.encode("utf-8"))
            pending.replace(path)
    return days, manifest, dataset.cache_dir.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.out.resolve())
