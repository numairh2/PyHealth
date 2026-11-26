"""Run the PMData baseline at different training label budgets.

    python -m examples.pmdata_label_efficiency DATA RUNS --fold 0 --seed 0 --label-pct 100 50 25 10

"""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.pmdata_full_pipeline import argument_parser as pipeline_parser, run as run_pipeline
from examples.pmdata_label_splits import LABEL_PCTS


def argument_parser():
    parser = pipeline_parser(__doc__)
    parser.add_argument("--label-pct", type=int, nargs="+", choices=LABEL_PCTS, default=LABEL_PCTS)
    return parser


def run(args):
    run_pipeline(args, label_pcts=args.label_pct)


if __name__ == "__main__":
    run(argument_parser().parse_args())
