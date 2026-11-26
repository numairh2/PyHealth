"""Tests for PMDataDataset, using a small fake PMData folder."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from pyhealth.datasets import PMDataDataset
from pyhealth.tasks import StressPredictionPMData

WELLNESS_HEADER = (
    "effective_time_frame,fatigue,mood,readiness,sleep_duration_h,"
    "sleep_quality,soreness,soreness_area,stress\n"
)


def write_json(path, records):
    with open(path, "w") as f:
        json.dump(records, f)


def write_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def make_participant(root, participant, wellness_text):
    """Makes one fake participant folder.

    Every participant has Fitbit data at 10:00 and 12:00 on 2019-11-01,
    and nothing at 11:00.
    """
    fitbit = os.path.join(root, participant, "fitbit")
    pmsys = os.path.join(root, participant, "pmsys")
    os.makedirs(fitbit)
    os.makedirs(pmsys)

    heart_rate = [
        {"dateTime": "2019-11-01 10:00:05", "value": {"bpm": 60, "confidence": 3}},
        {"dateTime": "2019-11-01 10:00:10", "value": {"bpm": 62, "confidence": 3}},
        {"dateTime": "2019-11-01 12:00:00", "value": {"bpm": 70, "confidence": 3}},
    ]
    steps = [
        {"dateTime": "2019-11-01 10:00:00", "value": "10"},
        {"dateTime": "2019-11-01 10:01:00", "value": "10"},
        {"dateTime": "2019-11-01 12:00:00", "value": "5"},
    ]
    calories = [{"dateTime": "2019-11-01 10:00:00", "value": "1.5"}]
    distance = [{"dateTime": "2019-11-01 10:00:00", "value": "100"}]

    write_json(os.path.join(fitbit, "heart_rate.json"), heart_rate)
    write_json(os.path.join(fitbit, "steps.json"), steps)
    write_json(os.path.join(fitbit, "calories.json"), calories)
    write_json(os.path.join(fitbit, "distance.json"), distance)
    write_text(os.path.join(pmsys, "wellness.csv"), WELLNESS_HEADER + wellness_text)


def make_overview(root):
    """Makes participant-overview.xlsx with a title row, like the real file.

    p02's gender has a hidden space, and p03 is missing on purpose.
    """
    rows = [
        ["PMData participants", None, None, None],
        ["Participant ID", "Age", "Height", "Gender"],
        ["p01", 48, 195, "male"],
        ["p02", 60, 180, "male\xa0"],
    ]
    path = os.path.join(root, "participant-overview.xlsx")
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


class TestPMDataDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.root = os.path.join(cls.temp_dir, "pmdata")
        os.makedirs(cls.root)

        # p01 reports twice on 2019-11-01 (stress 2, then 4)
        make_participant(
            cls.root,
            "p01",
            "2019-11-01T08:00:00.000Z,3,3,5,7,3,2,[],2\n"
            "2019-11-01T20:00:00.000Z,3,3,5,7,3,2,[],4\n"
            "2019-11-02T08:00:00.000Z,3,3,5,7,3,2,[],3\n",
        )
        make_participant(cls.root, "p02", "2019-11-01T08:00:00.000Z,3,3,5,7,3,2,[],1\n")
        make_participant(cls.root, "p03", "2019-11-01T08:00:00.000Z,3,3,5,7,3,2,[],5\n")
        make_overview(cls.root)

        # Build the dataset once for all tests. The cache goes in the temp folder.
        cache_dir = os.path.join(cls.temp_dir, "cache")
        cls.dataset = PMDataDataset(root=cls.root, cache_dir=cache_dir)

        cls.hourly = pd.read_csv(os.path.join(cls.root, "pmdata-hourly.csv"))
        cls.daily = pd.read_csv(os.path.join(cls.root, "pmdata-daily.csv"))
        cls.static = pd.read_csv(os.path.join(cls.root, "pmdata-static.csv"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def get_hour(self, patient_id, timestamp):
        """Returns one row of the hourly table."""
        rows = self.hourly[
            (self.hourly["patient_id"] == patient_id)
            & (self.hourly["timestamp"] == timestamp)
        ]
        return rows.iloc[0]

    def test_dataset_name(self):
        self.assertEqual(self.dataset.dataset_name, "pmdata")

    def test_patient_ids(self):
        self.assertEqual(sorted(self.dataset.unique_patient_ids), ["p01", "p02", "p03"])

    def test_processed_files_exist(self):
        for name in ["pmdata-hourly.csv", "pmdata-daily.csv", "pmdata-static.csv"]:
            self.assertTrue(os.path.exists(os.path.join(self.root, name)))

    def test_hourly_mean_and_sum(self):
        row = self.hourly[
            (self.hourly["patient_id"] == "p01")
            & (self.hourly["timestamp"] == "2019-11-01 10:00:00")
        ].iloc[0]

        self.assertEqual(row["heart_rate"], 61.0)
        self.assertEqual(row["steps"], 20.0)  # 10 + 10
        self.assertEqual(row["calories"], 1.5)
        self.assertEqual(row["distance"], 100.0)

    def test_missing_hour_is_not_filled(self):
        p01 = self.hourly[self.hourly["patient_id"] == "p01"]
        self.assertEqual(
            p01["timestamp"].tolist(), ["2019-11-01 10:00:00", "2019-11-01 12:00:00"]
        )

    def test_missing_value_stays_empty(self):
        # At 12:00 there are heart rate and steps, but no calories
        row = self.get_hour("p01", "2019-11-01 12:00:00")
        self.assertTrue(pd.isna(row["calories"]))

    def test_same_day_reports_keep_the_last(self):
        p01 = self.daily[self.daily["patient_id"] == "p01"]
        self.assertEqual(len(p01), 2)
        first_day = p01[p01["timestamp"] == "2019-11-01"].iloc[0]
        self.assertEqual(first_day["stress"], 4)

    def test_static_table(self):
        static = self.static.set_index("patient_id")
        self.assertEqual(static.loc["p01", "age"], 48)
        self.assertEqual(static.loc["p02", "gender"], "male")
        self.assertTrue(pd.isna(static.loc["p03", "age"]))

    def test_patient_events(self):
        patient = self.dataset.get_patient("p01")
        hourly = patient.get_events(event_type="pmdata_hourly")
        daily = patient.get_events(event_type="pmdata_daily")
        static = patient.get_events(event_type="pmdata_static")
        self.assertEqual(len(hourly), 2)
        self.assertEqual(len(daily), 2)
        self.assertEqual(len(static), 1)
        self.assertEqual(static[0].gender, "male")

    def test_unknown_patient_raises(self):
        with self.assertRaises(AssertionError):
            self.dataset.get_patient("p99")

    def test_stress_prediction_daily_samples(self):
        task = StressPredictionPMData(stress_threshold=4)
        samples = task(self.dataset.get_patient("p01"))

        # Hourly Fitbit data exists only on 2019-11-01; 2019-11-02 is skipped
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "p01")
        self.assertEqual(sample["record_id"], "2019-11-01")
        self.assertEqual(sample["stress"], 1)  # wellness stress 4 >= 4
        self.assertEqual(list(sample["heart_rate"]), [61.0, 70.0])
        self.assertEqual(list(sample["steps"]), [20.0, 5.0])
        self.assertEqual(sample["heart_rate"].dtype, np.dtype("float32"))
        # fatigue, mood, readiness, sleep_duration_h, sleep_quality, soreness
        self.assertEqual(list(sample["wellness"]), [3.0, 3.0, 5.0, 7.0, 3.0, 2.0])
        self.assertEqual(sample["wellness"].shape, (6,))
        # age, height, gender (male -> 0)
        self.assertEqual(list(sample["static"]), [48.0, 195.0, 0.0])
        self.assertEqual(sample["static"].shape, (3,))

        p02 = task(self.dataset.get_patient("p02"))
        self.assertEqual(len(p02), 1)
        self.assertEqual(p02[0]["stress"], 0)  # wellness stress 1 < 4
        self.assertEqual(list(p02[0]["static"]), [60.0, 180.0, 0.0])

        p03 = task(self.dataset.get_patient("p03"))
        self.assertEqual(len(p03), 1)
        self.assertTrue(np.isnan(p03[0]["static"]).all())  # missing overview row
    
    def test_text_present_in_sample(self):
        task = StressPredictionPMData(stress_threshold=4)
        samples = task(self.dataset.get_patient("p01"))
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(type(sample["text"]), str)
        self.assertTrue("stress" not in sample["text"])
        self.assertTrue("wellness" in sample["text"])
        self.assertTrue("static" in sample["text"])
        self.assertTrue("heart_rate" in sample["text"])
        self.assertTrue("steps" in sample["text"])
        self.assertTrue("calories" in sample["text"])
        self.assertTrue("distance" in sample["text"])
        self.assertTrue("wellness" in sample["text"])
        self.assertTrue("static" in sample["text"])
        self.assertTrue("heart_rate" in sample["text"])
        self.assertTrue("steps" in sample["text"])
        self.assertTrue("calories" in sample["text"])
        self.assertTrue("distance" in sample["text"])
    
   


class TestPMDataDatasetErrors(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            PMDataDataset(root=os.path.join(self.temp_dir, "does-not-exist"))

    def test_no_participants_raises(self):
        with self.assertRaises(ValueError):
            PMDataDataset(root=self.temp_dir)

    def test_wellness_missing_column_raises(self):
        make_participant(self.temp_dir, "p01", "")
        make_overview(self.temp_dir)
        # Replace wellness.csv with one that has no effective_time_frame column
        bad_file = os.path.join(self.temp_dir, "p01", "pmsys", "wellness.csv")
        write_text(bad_file, "fatigue,stress\n3,2\n")
        with self.assertRaises(ValueError):
            PMDataDataset(root=self.temp_dir)


if __name__ == "__main__":
    unittest.main()
