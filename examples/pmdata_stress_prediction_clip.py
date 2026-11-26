"""Run the PMData stress prediction experiment with the team's CLIP-style model.

Compare supervised training with contrastive pretraining followed by end-to-end
fine-tuning on the same fold, seed and labeled days. See the experiments and
original PMData test AUROCs in pmdata_stress_prediction_clip.md.

"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.pmdata_label_efficiency import argument_parser, run
from pyhealth.tasks.clip_pretraining import DAILY_FEATURES, HOURLY_FEATURES


def run_demo(output_parent: Path) -> None:
    """Run one CPU epoch per stage on two artificial participants, offline.

    Args:
        output_parent: Parent for a fresh directory of synthetic tables,
            a small random-weight BERT and the eight downstream results.
    """
    from transformers import BertConfig, BertModel

    output_parent.mkdir(parents=True, exist_ok=True)
    demo = Path(tempfile.mkdtemp(prefix="pmdata-demo-", dir=output_parent))
    data = demo / "data"
    data.mkdir()
    people = ("p01", "p02")
    with (data / "pmdata-static.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "age", "height", "gender"])
        writer.writerows([(people[0], 35, 160, "female"), (people[1], 42, 175, "male")])
    with (
        (data / "pmdata-hourly.csv").open("w", newline="", encoding="utf-8") as hourly,
        (data / "pmdata-daily.csv").open("w", newline="", encoding="utf-8") as daily,
    ):
        sensors, survey = csv.writer(hourly), csv.writer(daily)
        sensors.writerow(["patient_id", "timestamp", *HOURLY_FEATURES])
        survey.writerow(["patient_id", "timestamp", *DAILY_FEATURES, "stress"])
        for index, pid in enumerate(people):
            for day in range(1, 16):
                date = f"2020-01-{day:02}"
                for hour in (8, 18):
                    value = day + index + hour
                    sensors.writerow(
                        [
                            pid,
                            f"{date} {hour:02}:00:00",
                            60 + value,
                            4 * value,
                            0.5 * value,
                            0.01 * value,
                        ]
                    )
                # Ten labeled days, two surveys without stress, three Fitbit-only days.
                if day <= 12:
                    stress = 3 + (day + index) % 2 if day <= 10 else ""
                    survey.writerow([pid, date, 2, 3, 4, 7, 3, 2, stress])

    # The real encoder loads "bert-base-uncased". A tiny local model exercises
    # that same call in the demo without downloading a pretrained text encoder.
    bert = demo / "bert-base-uncased"
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "age",
        "height",
        "gender",
        "fatigue",
        "mood",
        "readiness",
        "heart",
        "rate",
        "steps",
        "calories",
        "distance",
        "mean",
        "is",
        "2",
        "3",
        "4",
        "7",
    ]
    config = BertConfig(
        vocab_size=len(tokens),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        BertModel(config).save_pretrained(bert)
    (bert / "vocab.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    print(f"Synthetic example directory: {demo.resolve()}", flush=True)
    args = argument_parser().parse_args(
        [
            str(data),
            str(demo / "runs"),
            "--fold",
            "0",
            "--seed",
            "0",
            "--pretrain-epochs",
            "1",
            "--supervised-epochs",
            "1",
            "--finetune-epochs",
            "1",
            "--batch-size",
            "8",
            "--text-batch-size",
            "8",
            "--device",
            "cpu",
            "--bert-dir",
            str(bert),
        ]
    )
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        run(args)
    finally:
        torch.set_num_threads(threads)


def main(argv: list[str] | None = None) -> None:
    """Run a full PMData experiment or its synthetic CPU example."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run an offline synthetic example.")
    demo.add_argument("output_parent", type=Path, help="Where to write the demo.")
    commands.add_parser(
        "run",
        parents=[argument_parser()],
        add_help=False,
        help="Run the baseline and label-budget comparison on real PMData.",
    )
    args = parser.parse_args(argv)
    if args.command == "demo":
        run_demo(args.output_parent)
    else:
        run(args)


if __name__ == "__main__":
    main()
