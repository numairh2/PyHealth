"""PMData dataset for PyHealth.

Author: Numair Hajyani
Paper: PMData: A Sports Logging Dataset (Thambawita et al., MMSys 2020)
Link: https://dl.acm.org/doi/10.1145/3339825.3394926
Description: Loads Fitbit and PMSys wellness data from 16 participants into
hourly, daily and static tables for stress detection.
"""

import json
import logging
import os

import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

WELLNESS_COLUMNS = [
    "fatigue",
    "mood",
    "readiness",
    "sleep_duration_h",
    "sleep_quality",
    "soreness",
    "stress",
]


class PMDataDataset(BaseDataset):
    """PMData: Fitbit and self-reported wellness data from 16 participants.

    This loader turns the raw PMData files into three tables:

    - ``pmdata_hourly``: heart rate (hourly mean) and steps, calories and
      distance (hourly sums). Hours with no data are left out, not set to 0.
    - ``pmdata_daily``: one wellness report per participant per day.
    - ``pmdata_static``: age, height and gender.

    Getting the data:
        - Source: https://datasets.simula.no/pmdata/
        - Access: open (CC BY 4.0), no account needed.
        - ``root``: the unzipped folder holding ``p01`` to ``p16`` and
          ``participant-overview.xlsx``.

    Fitbit times are local Norway time. Wellness times are UTC, so they are
    converted to Norway time (``Europe/Oslo``) before taking the date.

    Citation:
        Thambawita, V., Hicks, S. A., Borgli, H., et al. (2020). PMData: A
        Sports Logging Dataset. Proceedings of the 11th ACM Multimedia
        Systems Conference, 231-236.

    Args:
        root: Folder holding the unzipped PMData files.
        dataset_name: Name of the dataset. Defaults to ``"pmdata"``.
        config_path: Path to the YAML config. Defaults to ``configs/pmdata.yaml``.
        cache_dir: Where PyHealth caches its processed data.
        dev: If True, load only a small number of patients.
        refresh_processed: If True, rebuild the three CSVs even if they exist.

    Examples:
        >>> from pyhealth.datasets import PMDataDataset
        >>> dataset = PMDataDataset(root="/path/to/pmdata")
        >>> dataset.stats()
        >>> patient = dataset.get_patient("p01")
        >>> daily = patient.get_events(event_type="pmdata_daily")
        >>> print(daily[0].stress)
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        dev: bool = False,
        refresh_processed: bool = False,
    ) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"PMData folder not found: {root}")

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "pmdata.yaml"
            )

        hourly_csv = os.path.join(root, "pmdata-hourly.csv")
        daily_csv = os.path.join(root, "pmdata-daily.csv")
        static_csv = os.path.join(root, "pmdata-static.csv")
        csvs_exist = (
            os.path.exists(hourly_csv)
            and os.path.exists(daily_csv)
            and os.path.exists(static_csv)
        )
        if refresh_processed or not csvs_exist:
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["pmdata_hourly", "pmdata_daily", "pmdata_static"],
            dataset_name=dataset_name or "pmdata",
            config_path=config_path,
            cache_dir=cache_dir,
            dev=dev,
        )

    def find_participants(self, root: str) -> list[str]:
        """Finds the participant folders, such as ``p01`` to ``p16``.

        Args:
            root: Folder holding the unzipped PMData files.

        Returns:
            Sorted names of the folders that contain a ``fitbit`` folder.

        Raises:
            ValueError: If no participant folders are found.
        """
        participants = []
        for name in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, name, "fitbit")):
                participants.append(name)

        if len(participants) == 0:
            raise ValueError(f"No participant folders found in {root}")
        return participants

    def read_fitbit_file(self, path: str, how: str) -> pd.Series:
        """Reads one Fitbit JSON file and turns it into hourly values.

        Args:
            path: Path to a Fitbit file, such as ``p01/fitbit/steps.json``.
            how: ``"mean"`` to average each hour, or ``"sum"`` to add it up.

        Returns:
            One value per hour. Empty if the file doesn't exist.

        Raises:
            ValueError: If the file has no ``dateTime`` or ``value`` field.
        """
        if not os.path.exists(path):
            logger.warning(f"{path} not found, skipping it")
            return pd.Series(dtype=float)

        with open(path) as f:
            records = json.load(f)
        if len(records) == 0:
            return pd.Series(dtype=float)

        df = pd.DataFrame(records)
        if "dateTime" not in df.columns or "value" not in df.columns:
            raise ValueError(f"{path} is missing the dateTime or value field")

        # Heart rate values look like {"bpm": 54, "confidence": 3}
        if isinstance(df["value"][0], dict):
            df["value"] = df["value"].apply(lambda v: v["bpm"])

        # Other values are numbers stored as strings, such as "12"
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["dateTime"] = pd.to_datetime(df["dateTime"], format="%Y-%m-%d %H:%M:%S")
        values = df.set_index("dateTime")["value"]

        if how == "mean":
            return values.resample("h").mean()
        # min_count=1 keeps an hour with no records as NaN instead of 0
        return values.resample("h").sum(min_count=1)

    def build_hourly(self, root: str, participant: str) -> pd.DataFrame:
        """Builds the hourly table for one participant.

        Args:
            root: Folder holding the unzipped PMData files.
            participant: Participant folder name, such as ``"p01"``.

        Returns:
            One row per hour that has data in at least one column.
        """
        fitbit = os.path.join(root, participant, "fitbit")
        hourly = pd.DataFrame(
            {
                "heart_rate": self.read_fitbit_file(
                    os.path.join(fitbit, "heart_rate.json"), "mean"
                ),
                "steps": self.read_fitbit_file(
                    os.path.join(fitbit, "steps.json"), "sum"
                ),
                "calories": self.read_fitbit_file(
                    os.path.join(fitbit, "calories.json"), "sum"
                ),
                "distance": self.read_fitbit_file(
                    os.path.join(fitbit, "distance.json"), "sum"
                ),
            }
        )
        # Drop hours where every column is missing (no imputation)
        hourly = hourly.dropna(how="all")

        hourly["timestamp"] = hourly.index.strftime("%Y-%m-%d %H:%M:%S")
        hourly["patient_id"] = participant
        columns = ["patient_id", "timestamp", "heart_rate", "steps", "calories"]
        return hourly[columns + ["distance"]].reset_index(drop=True)

    def build_daily(self, root: str, participant: str) -> pd.DataFrame:
        """Builds the daily wellness table for one participant.

        Args:
            root: Folder holding the unzipped PMData files.
            participant: Participant folder name, such as ``"p01"``.

        Returns:
            One row per day. If a day has two reports, the later one is kept.

        Raises:
            FileNotFoundError: If ``wellness.csv`` is missing.
            ValueError: If ``wellness.csv`` is missing a column.
        """
        path = os.path.join(root, participant, "pmsys", "wellness.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Wellness file not found: {path}")

        wellness = pd.read_csv(path)
        for column in ["effective_time_frame"] + WELLNESS_COLUMNS:
            if column not in wellness.columns:
                raise ValueError(f"{path} is missing the column {column}")

        # Times are UTC, like "2019-11-01T08:31:40.751Z". Convert to Norway time.
        times = pd.to_datetime(wellness["effective_time_frame"], utc=True)
        times = times.dt.tz_convert("Europe/Oslo")
        wellness["timestamp"] = times.dt.strftime("%Y-%m-%d")

        # Sort by time, then keep the last report of each day
        wellness = wellness.sort_values("effective_time_frame")
        wellness = wellness.drop_duplicates(subset="timestamp", keep="last")

        wellness["patient_id"] = participant
        columns = ["patient_id", "timestamp"] + WELLNESS_COLUMNS
        return wellness[columns].reset_index(drop=True)

    def build_static(self, root: str, participants: list[str]) -> pd.DataFrame:
        """Builds the static table from ``participant-overview.xlsx``.

        Args:
            root: Folder holding the unzipped PMData files.
            participants: Participant IDs. Each one gets a row.

        Returns:
            One row per participant with age, height and gender.

        Raises:
            FileNotFoundError: If ``participant-overview.xlsx`` is missing.
            ValueError: If the spreadsheet is missing a column.
        """
        path = os.path.join(root, "participant-overview.xlsx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Participant overview not found: {path}")

        # Row 0 is a title row, so the real column names are on row 1
        overview = pd.read_excel(path, header=1)
        for column in ["Participant ID", "Age", "Height", "Gender"]:
            if column not in overview.columns:
                raise ValueError(f"{path} is missing the column {column}")

        static = pd.DataFrame()
        static["patient_id"] = overview["Participant ID"].astype(str).str.strip()
        static["age"] = overview["Age"]
        static["height"] = overview["Height"]
        # One gender value has a hidden space ("male\xa0"), so strip it
        static["gender"] = overview["Gender"].str.strip().str.lower()

        # Left join so every participant gets a row, even with no overview entry
        all_participants = pd.DataFrame({"patient_id": participants})
        return all_participants.merge(static, on="patient_id", how="left")

    def prepare_metadata(self, root: str) -> None:
        """Builds the hourly, daily and static CSVs and saves them in ``root``.

        Args:
            root: Folder holding the unzipped PMData files.
        """
        participants = self.find_participants(root)
        logger.info(f"Building PMData tables for {len(participants)} participants")

        hourly_tables = []
        daily_tables = []
        for participant in participants:
            hourly_tables.append(self.build_hourly(root, participant))
            daily_tables.append(self.build_daily(root, participant))

        hourly = pd.concat(hourly_tables, ignore_index=True)
        daily = pd.concat(daily_tables, ignore_index=True)
        static = self.build_static(root, participants)

        hourly.to_csv(os.path.join(root, "pmdata-hourly.csv"), index=False)
        daily.to_csv(os.path.join(root, "pmdata-daily.csv"), index=False)
        static.to_csv(os.path.join(root, "pmdata-static.csv"), index=False)
        logger.info(f"Saved {len(hourly)} hourly, {len(daily)} daily rows")
