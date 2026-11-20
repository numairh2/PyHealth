"""Daily stress prediction on PMData Fitbit and wellness reports.

Each sample is one calendar day with a wellness report. That day's hourly
Fitbit series (heart rate, steps, calories, distance) are returned as
float tensors, and the other daily wellness fields (excluding stress) as a
tabular vector. Hours with no Fitbit data are omitted, matching
:class:`~pyhealth.datasets.PMDataDataset`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

import math
import numpy as np

from .base_task import BaseTask

HOURLY_FEATURES = ("heart_rate", "steps", "calories", "distance")
DAILY_FEATURES = (
    "fatigue",
    "mood",
    "readiness",
    "sleep_duration_h",
    "sleep_quality",
    "soreness",
)
STATIC_FEATURES = ("age", "height", "gender")

# Encoded into the static tensor; unknown / missing stays NaN.
_GENDER_TO_FLOAT = {"male": 0.0, "female": 1.0}


def _as_float(value: Any) -> float:
    """Converts an event attribute to float, or NaN if missing."""
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _static_value(name: str, value: Any) -> float:
    """Converts a static attribute for the ``static`` tensor."""
    if name == "gender":
        if value is None:
            return float("nan")
        return _GENDER_TO_FLOAT.get(str(value).strip().lower(), float("nan"))
    return _as_float(value)


class StressPredictionPMData(BaseTask):
    """Predict daily stress from that day's Fitbit series and wellness fields.

    Works with :class:`~pyhealth.datasets.PMDataDataset`.

    For each ``pmdata_daily`` wellness report this task:

    1. collects ``pmdata_hourly`` events on the same calendar day
    2. builds one float vector per hourly channel (heart rate, steps,
       calories, distance), sorted by time
    3. packs the other daily wellness fields (fatigue, mood, readiness,
       sleep duration/quality, soreness) into a ``wellness`` vector;
       ``stress`` is not included there
    4. packs patient ``static`` features (age, height, gender) from
       ``pmdata_static``; gender is ``0`` male / ``1`` female
    5. binarizes ``stress`` with ``stress >= stress_threshold`` as the
       positive class (PMSys stress is typically 1-5)

    Days with no hourly Fitbit rows, or with a missing stress value, are
    skipped. Hours present for the day keep NaN in channels that were not
    recorded that hour.

    Args:
        stress_threshold: Stress at or above this value is labeled 1.
            Default is 4.

    Examples:
        >>> from pyhealth.datasets import PMDataDataset
        >>> from pyhealth.tasks import StressPredictionPMData
        >>> dataset = PMDataDataset(root="/path/to/pmdata")  # doctest: +SKIP
        >>> samples = dataset.set_task(StressPredictionPMData())  # doctest: +SKIP
        >>> samples[0]["heart_rate"].shape  # doctest: +SKIP
        >>> samples[0]["wellness"].shape  # doctest: +SKIP
        >>> samples[0]["static"].shape  # doctest: +SKIP
    """

    task_name: str = "StressPredictionPMData"
    input_schema: ClassVar[dict[str, str]] = {
        "heart_rate": "tensor",
        "steps": "tensor",
        "calories": "tensor",
        "distance": "tensor",
        "wellness": "tensor",
        "static": "tensor",
        "texts": "text",
    }
    output_schema: ClassVar[dict[str, str]] = {"stress": "binary"}

    def __init__(self, stress_threshold: float = 4.0) -> None:
        super().__init__()
        self.stress_threshold = float(stress_threshold)

    def get_time_series_stats(self, sample, time_series_columns):
        from scipy import stats

        time_series_descriptions = ""
        for column in time_series_columns:
            values = np.asarray(sample[column], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            time_series_descriptions += f"{column}"
            time_series_descriptions += f"mean is {np.mean(values)},"
            time_series_descriptions += f"std is {np.std(values)},"
            time_series_descriptions += f"min is {np.min(values)},"
            time_series_descriptions += f"max is {np.max(values)},"
            time_series_descriptions += f"25th percentile is {np.percentile(values, 25)},"
            time_series_descriptions += f"median is {np.median(values)},"
            time_series_descriptions += f"75th percentile is {np.percentile(values, 75)},"
            time_series_descriptions += f"root mean square is {np.sqrt(np.mean(values ** 2))},"
            time_series_descriptions += f"kurtosis is {stats.kurtosis(values)},"
            time_series_descriptions += f"skew is {stats.skew(values)},"
            time_series_descriptions += f"interquartile range is {stats.iqr(values)}\n"
        return time_series_descriptions

    def prompt_generation(self, sample):
        description = ""
        for i, feature in enumerate(STATIC_FEATURES):
            if not np.isnan(sample["static"][i]):
                description += f"{feature}: {sample['static'][i]}\n".lower()
        for i, feature in enumerate(DAILY_FEATURES):
            if not np.isnan(sample["wellness"][i]):
                description += f"{feature}: {sample['wellness'][i]}\n".lower()

        description += self.get_time_series_stats(sample, HOURLY_FEATURES)
        sample["texts"] = description
        return sample

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per daily wellness report with hourly tensors."""
        hourly_by_day: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {name: [] for name in HOURLY_FEATURES}
        )

        for event in patient.get_events(event_type="pmdata_hourly"):
            if event.timestamp is None:
                continue
            day = event.timestamp.strftime("%Y-%m-%d")
            day_series = hourly_by_day[day]
            for name in HOURLY_FEATURES:
                day_series[name].append(_as_float(getattr(event, name, None)))

        static_events = patient.get_events(event_type="pmdata_static")
        static_event = static_events[0] if static_events else None
        static = np.asarray(
            [
                _static_value(
                    name,
                    None if static_event is None else getattr(static_event, name, None),
                )
                for name in STATIC_FEATURES
            ],
            dtype=np.float32,
        )

        samples: list[dict[str, Any]] = []
        for report in patient.get_events(event_type="pmdata_daily"):
            if report.timestamp is None:
                continue
            day = report.timestamp.strftime("%Y-%m-%d")
            stress = _as_float(getattr(report, "stress", None))
            if not math.isfinite(stress):
                continue

            day_series = hourly_by_day.get(day)
            if day_series is None or len(day_series["heart_rate"]) == 0:
                continue

            sample: dict[str, Any] = {
                "patient_id": patient.patient_id,
                "record_id": day,
                "stress": int(stress >= self.stress_threshold),
                "wellness": np.asarray(
                    [
                        _as_float(getattr(report, name, None))
                        for name in DAILY_FEATURES
                    ],
                    dtype=np.float32,
                ),
                "static": static.copy(),
            }
            for name in HOURLY_FEATURES:
                sample[name] = np.asarray(day_series[name], dtype=np.float32)

            # prompt generation
            sample = self.prompt_generation(sample)
            samples.append(sample)

        return samples
