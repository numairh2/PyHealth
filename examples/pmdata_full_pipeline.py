"""Run the PMData baseline: contrastive pretraining, supervised training and fine-tuning.

    python -m examples.pmdata_full_pipeline DATA RUNS --fold 0 --seed 0

Each fold/seed shares one pretrain.pt. Each downstream run saves its AUROCs in run.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from uuid import uuid4

import torch
from filelock import FileLock

if not __package__:  # use this checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.pmdata_label_splits import FOLDS, SEEDS, build_fold, prepare
from examples.pmdata_training import classify, pretrain


def _check_cached_config(path, saved, expected):
    """Match the core settings for this checkpoint or downstream result."""
    previous = saved if isinstance(saved, dict) else {}
    core_fields = ("fold", "seed", "lr", "batch_size", "epochs", "label_pct")
    changed = [
        field for field in core_fields
        if field in expected and previous.get(field) != expected[field]
    ]
    if changed:
        raise ValueError(f"Cached settings differ for {path} ({', '.join(changed)}). Use a separate run folder.")


def pretraining_checkpoint(path, data, args, seed, device, settings):
    config = {
        **settings, "epochs": args.pretrain_epochs, "text_batch_size": args.text_batch_size,
        "text_model": str(args.bert_dir or "bert-base-uncased"),
    }
    if path.is_file():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _check_cached_config(path, checkpoint.get("config"), config)
        return checkpoint
    if args.stage == "finetune":
        raise FileNotFoundError(f"Run --stage pretrain first to create {path}.")
    checkpoint = {"config": config, "id": uuid4().hex, "state_dict": pretrain(data, args, seed, device)}
    pending = path.with_suffix(".partial")
    torch.save(checkpoint, pending)
    pending.replace(path)
    return checkpoint


def run_pair(data, args, fold, seed, dataset_id, device, label_pcts):
    base = args.out / f"fold_{fold}" / f"seed_{seed}"
    base.mkdir(parents=True, exist_ok=True)
    settings = {
        "dataset_id": dataset_id, "fold": fold, "seed": seed,
        "device": str(device), "lr": args.lr, "batch_size": args.batch_size,
    }
    with FileLock(str(base / "training.lock")):
        checkpoint = None
        if args.stage != "supervised":
            checkpoint = pretraining_checkpoint(base / "pretrain.pt", data, args, seed, device, settings)
        if args.stage == "pretrain":
            return

        methods = ("supervised", "finetune") if args.stage == "all" else (args.stage,)
        for pct in label_pcts:
            indices = data.label_indices(seed, pct)  # Shared by the two methods, supervised and finetune.
            for method in methods:
                tuned = method == "finetune"
                config = {**settings, "method": method, "label_pct": pct,
                          "epochs": args.finetune_epochs if tuned else args.supervised_epochs}
                path = base / f"{method}_{pct}" / "run.json"
                result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
                if result is not None:
                    _check_cached_config(path, result.get("config"), config)
                else:
                    weights = checkpoint["state_dict"] if tuned else None
                    scores = classify(data, indices, args, seed, device, weights)
                    result = {
                        "config": config, "pretrain_id": checkpoint.get("id") if tuned else None, **scores,
                        "train_days": len(indices), "val_days": len(data.val), "test_days": len(data.test),
                    }
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pending = path.with_suffix(".partial")
                    pending.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    pending.replace(path)
                print(f"fold={fold} seed={seed} {method} labels={pct}% "
                      f"val_AUROC={result['val_auroc']:.4f} test_AUROC={result['test_auroc']:.4f}")


def run(args, label_pcts=(100,)):
    if min(args.pretrain_epochs, args.supervised_epochs, args.finetune_epochs, args.text_batch_size) < 1:
        raise ValueError("Epoch counts and text batch size must be positive.")
    if args.batch_size < 2 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Use a batch size >= 2 and a positive, finite learning rate.")
    args.root, args.out = args.root.resolve(), args.out.resolve()
    local_bert = args.root.parent / "bert-base-uncased"
    if args.bert_dir is None and local_bert.is_dir():
        args.bert_dir = local_bert
    if args.bert_dir is not None:
        args.bert_dir = args.bert_dir.resolve()
        if args.bert_dir.name != "bert-base-uncased" or not (args.bert_dir / "config.json").is_file():
            raise ValueError("--bert-dir must point to a local bert-base-uncased folder.")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    days, manifest, dataset_id = prepare(args.root, args.out)
    for fold in args.fold:
        data = build_fold(days, manifest.loc[manifest.fold.eq(fold)])
        for seed in args.seed:
            run_pair(data, args, fold, seed, dataset_id, device, label_pcts)


def argument_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="PMData folder")
    parser.add_argument("out", type=Path, help="Experiment output folder")
    parser.add_argument("--stage", choices=("all", "pretrain", "supervised", "finetune"), default="all")
    parser.add_argument("--fold", type=int, nargs="+", choices=FOLDS, default=FOLDS)
    parser.add_argument("--seed", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    parser.add_argument("--pretrain-epochs", type=int, default=300)
    parser.add_argument("--supervised-epochs", type=int, default=300)
    parser.add_argument("--finetune-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", help="Defaults to CUDA when available")
    parser.add_argument("--bert-dir", type=Path, help="Local bert-base-uncased folder")
    return parser


if __name__ == "__main__":
    run(argument_parser().parse_args())
