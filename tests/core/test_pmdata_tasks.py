"""PMData Task tests using one synthetic PyHealth patient.

"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import clip_pretraining
from pyhealth.tasks import PMDataPretraining, StressPredictionPMData
from pyhealth.tasks.clip_pretraining import HOURLY_FEATURES


def survey_patient(stress: int | None, *, same_day_survey: bool = True) -> Patient:
    """Build two sensor hours, a same-day survey and a separate survey-only day."""
    records = []
    for hour, pulse, calories in [(10, 61, 1.5), (12, 70, None)]:
        records.append(
            {
                "event_type": "pmdata_hourly",
                "timestamp": datetime(2020, 1, 1, hour),
                "pmdata_hourly/heart_rate": pulse,
                "pmdata_hourly/steps": hour,
                "pmdata_hourly/calories": calories,
                "pmdata_hourly/distance": hour / 10,
            }
        )
    survey = {
        "event_type": "pmdata_daily",
        "pmdata_daily/fatigue": 3,
        "pmdata_daily/mood": 3,
        "pmdata_daily/readiness": 5,
        "pmdata_daily/sleep_duration_h": 7,
        "pmdata_daily/sleep_quality": 3,
        "pmdata_daily/soreness": 2,
    }
    if same_day_survey:
        records.append(
            {**survey, "timestamp": datetime(2020, 1, 1), "pmdata_daily/stress": stress}
        )
    records.append(
        {**survey, "timestamp": datetime(2020, 1, 2), "pmdata_daily/stress": 5}
    )
    for row in records:
        row["patient_id"] = "p01"
    return Patient("p01", pl.DataFrame(records))


@pytest.mark.parametrize(("score", "expected"), [(3, 0), (4, 1), (None, None)])
def test_stress_labels_require_a_score_and_fitbit(
    score: int | None,
    expected: int | None,
) -> None:
    """The threshold includes four, and a missing score yields no sample."""
    samples = StressPredictionPMData()(survey_patient(score))
    assert [(s["record_id"], s["stress"]) for s in samples] == (
        [] if expected is None else [("2020-01-01", expected)]
    )


def test_missing_stress_preserves_wellness_and_label_free_prompt() -> None:
    """Changing stress alone leaves the contrastive inputs and prompt identical."""
    task = PMDataPretraining()
    (reference,) = task(survey_patient(4))
    (sample,) = task(survey_patient(None))
    assert "stress" not in sample and "stress" not in reference
    assert (sample["patient_id"], sample["record_id"]) == ("p01", "2020-01-01")
    heart_rate = np.full(24, np.nan, dtype=np.float32)
    heart_rate[[10, 12]] = [61, 70]
    calories = np.full(24, np.nan, dtype=np.float32)
    calories[10] = 1.5
    np.testing.assert_array_equal(sample["heart_rate"], heart_rate)
    np.testing.assert_array_equal(sample["calories"], calories)
    np.testing.assert_array_equal(sample["wellness"], [3, 3, 5, 7, 3, 2])
    np.testing.assert_array_equal(sample["static"], np.full(3, np.nan))
    assert "fatigue" in sample["texts"] and "heart_rate" in sample["texts"]
    assert sample["texts"] == reference["texts"]


def test_survey_on_another_day_is_not_joined() -> None:
    """A Fitbit-only day remains eligible for pretraining with six missing items."""
    patient = survey_patient(None, same_day_survey=False)
    (sample,) = PMDataPretraining()(patient)
    assert sample["record_id"] == "2020-01-01"
    np.testing.assert_array_equal(sample["wellness"], np.full(6, np.nan))
    assert StressPredictionPMData()(patient) == []


def test_pretraining_encoding_is_independent_of_other_days_in_batch(monkeypatch):
    task = PMDataPretraining()
    (sparse,) = task(survey_patient(None))
    hours = list(range(24))
    full_day = pl.DataFrame({
        "patient_id": ["p02"] * 24,
        "event_type": ["pmdata_hourly"] * 24,
        "timestamp": [datetime(2020, 1, 1, hour) for hour in hours],
        **{f"pmdata_hourly/{field}": hours for field in HOURLY_FEATURES},
    })
    (dense,) = task(Patient("p02", full_day))
    dataset = create_sample_dataset([sparse, dense], task.input_schema, task.output_schema)
    solo = next(iter(get_dataloader(dataset, 1)))
    paired = next(iter(get_dataloader(dataset, 2)))
    for field in HOURLY_FEATURES:
        assert solo[field].shape == (1, 24)
        torch.testing.assert_close(solo[field][0], paired[field][0], equal_nan=True)

    # encode() uses the real wearable and tabular networks. Its unused BERT branch stays offline.
    monkeypatch.setattr(
        clip_pretraining,
        "TextEmbedding",
        lambda **kwargs: torch.nn.ModuleDict({"fc": torch.nn.Identity()}),
    )
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(0)
        model = clip_pretraining.ClipMultimodalEncoderPretraining(dataset).eval()
        torch.testing.assert_close(model.encode(**solo)[0], model.encode(**paired)[0])


def test_pretraining_rejects_duplicate_clock_hours():
    patient = survey_patient(None)
    duplicate = patient.data_source.filter(pl.col("event_type") == "pmdata_hourly").head(1)
    patient = Patient("p01", pl.concat([patient.data_source, duplicate]))
    with pytest.raises(ValueError, match="Duplicate Fitbit hour for p01 on 2020-01-01"):
        PMDataPretraining()(patient)
