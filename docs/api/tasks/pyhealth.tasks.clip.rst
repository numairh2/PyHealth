pyhealth.tasks.clip
=========================================

Daily stress prediction on PMData Fitbit and PMSys wellness reports.

Each sample is one calendar day with a wellness report. The task pairs that
day's hourly Fitbit series (heart rate, steps, calories, distance) with
tabular wellness fields (fatigue, mood, readiness, sleep, soreness) and
static demographics (age, height, gender), and builds a short text prompt
from the same features. The label is binary stress
(``stress >= stress_threshold``; default threshold 4). Days without hourly
Fitbit data are skipped.

Task parameters(StressPredictionPMData):

- stress_threshold: float, default=4.0

.. autoclass:: pyhealth.tasks.clip_pretraining.StressPredictionPMData
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.clip_pretraining.PMDataPretraining
    :members:
    :undoc-members:
    :show-inheritance: