"""Check PMData experiment splits and results with two synthetic participants."""

import json
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from examples import pmdata_full_pipeline as pipeline
from examples import pmdata_label_efficiency as ablation
from examples.pmdata_label_splits import HOURLY_FEATURES, UserDays, make_manifest


# Distinct stub scores make a swapped method or validation/test result visible.
SCORES = {"supervised": (0.4, 0.5), "finetune": (0.7, 0.6)}


def day_ids(samples):
    return {(sample["patient_id"], sample["record_id"]) for sample in samples}


def run_result(args, method, pct):
    path = args.out / "fold_0" / "seed_0" / f"{method}_{pct}" / "run.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    # Start from task outputs; test_pmdata.py covers raw data loading and parsing.
    days = UserDays(features={}, labels={})
    for patient in ("p01", "p02"):
        for day in range(1, 16):
            date = f"2019-11-{day:02d}"
            days.features[patient, date] = {
                "patient_id": patient, "record_id": date, "texts": "Fitbit summary",
                "wellness": np.full(6, 3.0 if day <= 10 else np.nan),
                "static": np.array([31.0, 170.0, 0.0]),
                **{key: np.full(24, float(day)) for key in HOURLY_FEATURES},
            }
            if day <= 10:  # Five positive and five negative days per person.
                days.labels[patient, date] = int(day % 2 == 0)
    monkeypatch.setattr(
        pipeline, "prepare", lambda root, out: (days, make_manifest(days), "synthetic")
    )

    pretrain = Mock(return_value={"weight": torch.tensor(3.0)})
    runs = {"supervised": [], "finetune": []}

    def classify(data, indices, args, seed, device, weights):
        method = "supervised" if weights is None else "finetune"
        if weights is not None:
            assert torch.equal(weights["weight"], pretrain.return_value["weight"])
        train = day_ids(data.train[i] for i in indices)
        runs[method].append((train, day_ids(data.val), day_ids(data.test)))
        val, test = SCORES[method]
        return {"val_auroc": val, "test_auroc": test}

    monkeypatch.setattr(pipeline, "pretrain", pretrain)
    monkeypatch.setattr(pipeline, "classify", classify)
    cli = [
        str(tmp_path), str(tmp_path / "runs"), "--fold", "0", "--seed", "0",
        "--device", "cpu",
    ]
    return cli, pretrain, runs


def test_full_pipeline_uses_all_training_labels(experiment):
    cli, pretrain, runs = experiment
    args = pipeline.argument_parser().parse_args(cli)
    pipeline.run(args)

    pretrain.assert_called_once()
    fold = pretrain.call_args.args[0]
    train, val, test = day_ids(fold.train), day_ids(fold.val), day_ids(fold.test)
    pool = day_ids(fold.pretrain)
    assert (len(train), len(val), len(test)) == (14, 2, 4)
    assert train == {key for key in pool if key[1] <= "2019-11-10"}
    assert len(pool - train) == 8  # Four unlabeled training days per person.
    assert all("stress" not in sample for sample in fold.pretrain)
    assert pool.isdisjoint(val | test)
    assert val.isdisjoint(test)
    assert runs["supervised"] == runs["finetune"] == [(train, val, test)]
    for method, scores in SCORES.items():
        result = run_result(args, method, 100)
        assert result["train_days"] == len(train)
        assert (result["val_auroc"], result["test_auroc"]) == scores


def test_label_efficiency_pairs_nested_subsets(experiment):
    cli, pretrain, runs = experiment
    args = ablation.argument_parser().parse_args(cli)
    # Both entry points share the checkpoint and the completed 100% baseline.
    pipeline.run(pipeline.argument_parser().parse_args(cli))
    ablation.run(args)

    pretrain.assert_called_once()
    fold = pretrain.call_args.args[0]
    assert runs["supervised"] == runs["finetune"]
    selected = {}
    held_out = day_ids(fold.val), day_ids(fold.test)
    # Seven labels of each class remain in training. Each budget rounds by class.
    counts = {10: 2, 25: 4, 50: 8, 100: 14}
    assert set(args.label_pct) == set(counts)
    by_size = sorted(runs["finetune"], key=lambda run: len(run[0]))
    for (pct, count), (train, val, test) in zip(counts.items(), by_size, strict=True):
        selected[pct] = train
        assert len(train) == count
        assert sum(int(date[-2:]) % 2 == 0 for _, date in train) == len(train) // 2
        assert (val, test) == held_out
        for method in SCORES:
            assert run_result(args, method, pct)["train_days"] == len(train)
    assert selected[10] < selected[25] < selected[50] < selected[100]
    assert selected[100] == day_ids(fold.train)


def test_cache_reuses_results_across_devices_and_legacy_metadata(experiment):
    cli, pretrain, runs = experiment
    parser = pipeline.argument_parser()
    args = parser.parse_args([*cli, "--device", "cuda:0"])
    pipeline.run(args)
    path = args.out / "fold_0" / "seed_0" / "pretrain.pt"
    checkpoint = torch.load(path, weights_only=True)
    original = run_result(args, "finetune", 100)
    assert original["pretrain_id"] == checkpoint["id"]
    checkpoint["config"].update(model={"implementation_sha256": "old"}, preprocessing_sha256="old")
    checkpoint["id"] = "replacement-checkpoint"
    torch.save(checkpoint, path)

    pipeline.run(parser.parse_args([*cli, "--device", "cuda:1", "--text-batch-size", "4"]))
    pretrain.assert_called_once()
    assert len(runs["supervised"]) == len(runs["finetune"]) == 1
    assert run_result(args, "finetune", 100) == original


@pytest.mark.parametrize(("stage", "option", "value", "field"), [
    ("pretrain", "--lr", "0.002", "lr"),
    ("pretrain", "--pretrain-epochs", "2", "epochs"),
    ("supervised", "--batch-size", "8", "batch_size"),
    ("supervised", "--supervised-epochs", "2", "epochs"),
    ("finetune", "--finetune-epochs", "2", "epochs"),
])
def test_cache_rejects_changed_training_settings(experiment, stage, option, value, field):
    cli, pretrain, runs = experiment
    parser = pipeline.argument_parser()
    pipeline.run(parser.parse_args(cli))
    with pytest.raises(ValueError, match=rf"\({field}\)"):
        pipeline.run(parser.parse_args([*cli, "--stage", stage, option, value]))
    pretrain.assert_called_once()
    assert len(runs["supervised"]) == len(runs["finetune"]) == 1


@pytest.mark.parametrize("field", ["fold", "seed", "label_pct"])
def test_cached_result_requires_split_and_budget(experiment, field):
    cli, _, runs = experiment
    args = pipeline.argument_parser().parse_args([*cli, "--stage", "supervised"])
    pipeline.run(args)
    path = args.out / "fold_0" / "seed_0" / "supervised_100" / "run.json"
    result = run_result(args, "supervised", 100)
    result["config"].pop(field)
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"\({field}\)"):
        pipeline.run(args)
    assert len(runs["supervised"]) == 1
