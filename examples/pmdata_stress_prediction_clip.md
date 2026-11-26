# PMData daily stress classification and label efficiency

This example compares supervised training from scratch with contrastive
pretraining followed by end-to-end fine-tuning. The question is whether
pretraining changes held-out test AUROC when fewer training labels are
available. It uses the team's PMData loader and CLIP-style encoder. The
original method is described by [Yang, Yu and Sano (CHIL 2025)](https://proceedings.mlr.press/v287/yang25a.html).

## Running the example

From the repository root, use Python 3.12 or 3.13 with PyTorch installed for
your CPU or GPU. Install the checkout and the Excel reader used by the raw
PMData loader.

```console
python -m pip install -e . openpyxl pytest
python -m examples.pmdata_stress_prediction_clip demo runs/pmdata-demo
```

The demo creates two synthetic participants and a tiny, randomly initialized
local BERT. It runs the actual pretraining and classification code on CPU for
one epoch per stage, fold 0, seed 0 and all four label budgets. Its eight test
scores verify execution. The table below uses the real PMData runs.

For real data, unpack PMData into `data/pmdata`, with the 16 participant folders
and `participant-overview.xlsx` inside. The text encoder loads
`bert-base-uncased`. Use the available Hugging Face weights or supply a local
folder named `bert-base-uncased` with `--bert-dir PATH`. [PMData dataset](https://datasets.simula.no/pmdata/)

```console
python -m examples.pmdata_stress_prediction_clip run data/pmdata runs/pmdata-5fold --device cuda:0 --label-pct 100
python -m examples.pmdata_stress_prediction_clip run data/pmdata runs/pmdata-5fold --device cuda:0 --label-pct 50 25 10
```

Both commands use folds 0–4 and seeds 0 and 1 by default. For one pair, add
`--fold 0 --seed 0`. The original `examples.pmdata_full_pipeline` and
`examples.pmdata_label_efficiency` entry points accept the same training
settings. The baseline entry point uses all available training labels.

The output directory contains `manifest.csv`, one `fold_F/seed_S/pretrain.pt`
per fold and seed, and one `fold_F/seed_S/METHOD_PCT/run.json` per downstream
run. The JSON records the configuration, sample counts, validation AUROC and
test AUROC. To reuse a checkpoint or result, the fold, seed, learning rate,
training batch size and epoch count for that stage must match. Downstream
results also require the same label percentage. The device and pretraining
checkpoint ID are recorded for reference. Use a fresh output directory to
evaluate changes to the data, preprocessing, training code or model weights.

## Tasks and evaluation protocol

- **Eligible user-days.** The dataset yielded 2,395 Fitbit days from 16
  participants. Of these, 1,723 had an observed stress score and 672 had
  none. The positive class is defined as an observed score of 4 or higher
  (298 positive and 1,425 negative days). Classification requires both Fitbit
  data and an observed same-day score. Pretraining keeps Fitbit days with or
  without a survey and excludes stress from the text prompt and input schema.
- **Features.** Each day has four hourly Fitbit channels (heart rate, steps,
  calories and distance), six wellness items and three static fields. The
  public `pyhealth.tasks.PMDataPretraining` task places each observation in
  its clock-hour position from 0 through 23. Numeric-feature quantile
  transforms are fitted on training-day observations within each fold.
  The integrated encoder fills missing hourly values with 0 and missing
  tabular values with −1.
- **Splits and label budgets.** Five participant-mixed folds hold out different
  user-days for testing. Within each fold, about 10% of the remaining eligible
  labeled days are reserved for validation. Participants can appear on both
  sides on different dates. Training labels are sampled within class using
  nested 10%, 25%, 50% and 100% subsets. The two methods receive identical
  labeled days for each fold, seed and budget. Validation and test days stay
  fixed across budgets and are excluded from pretraining.
- **Training.** Each fold and seed has one contrastive checkpoint trained on
  all its eligible training days, ignoring labels. Each budget initializes a
  separate fine-tune from that checkpoint. Direct supervision starts from
  random weights. The runs use Adam, learning rate 0.001 and batch size 512.
  Direct supervision and pretraining each use 300 epochs, and fine-tuning uses
  150. The classifier with the highest validation AUROC is evaluated on the
  held-out test days.

## Original PMData results

Each row uses 10 matched fold–seed pairs. AUROCs are on the 0–1 scale. Values
are arithmetic means ± sample standard deviations across the ten run values.
For each pair, Δ is fine-tuning test AUROC minus supervised test AUROC.
Fine-tuning wins counts strictly positive paired differences.

| Training labels | Supervised from scratch | Pretraining + fine-tuning | Paired Δ | Fine-tuning wins |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 0.6943 ± 0.0694 | 0.7270 ± 0.0426 | +0.0327 ± 0.0911 | 7/10 |
| 25% | 0.7340 ± 0.0655 | 0.7638 ± 0.0339 | +0.0298 ± 0.0763 | 7/10 |
| 50% | 0.7985 ± 0.0173 | 0.7882 ± 0.0361 | −0.0103 ± 0.0414 | 5/10 |
| 100% | 0.8250 ± 0.0186 | 0.8065 ± 0.0420 | −0.0185 ± 0.0406 | 4/10 |

Pretraining led on average at 10% and 25% of labels. The mean paired
advantage grew by 0.0513 AUROC from 100% to 10%, and the low-label advantage
varied by fold and seed. The SDs describe run-to-run variation on these five
splits. They include two seeds on each fold.

At full labels, Table 1 of the paper reports PMData AUROCs of 0.8081 for
supervised training and 0.8114 for the self-supervised method. Our full-label
means put supervised training ahead. The paper uses four seeds, a three-layer,
four-head time-series encoder and global normalization. Our runs use two
seeds, the team's one-layer, one-head encoder and fold-specific training
normalization. Table 1 also tunes hyperparameters per fold and seed, whereas
this experiment fixes them across label budgets. Our ablation varies downstream
labels while holding pretraining data fixed. Paper Table 5 varies pretraining
data with a fixed labeled set.
These participant-mixed results describe performance on held-out days.
[Original paper, Tables 1 and 5](https://proceedings.mlr.press/v287/yang25a.html)

## Focused tests

```console
python -m pytest -q tests/core/test_pmdata_tasks.py tests/core/test_pmdata_experiments.py
```

The task tests use one in-memory patient. They check the label boundary,
missing scores, wellness retention and prompt independence. The experiment
tests use two synthetic participants and stub training to check the shared
pretraining pool, held-out days, paired budgets and saved metrics.
