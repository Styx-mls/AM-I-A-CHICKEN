from pathlib import Path

from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.suites import train_test_validation


TRAINING_DIR = Path(__file__).resolve().parent

PROCESSED_DATA_DIR = (
    TRAINING_DIR
    / "Core_Dataset"
    / "processed_v1"
)


def run_deepchecks():

    train_data, test_data = classification_dataset_from_directory(
        root=str(PROCESSED_DATA_DIR),
        object_type="VisionData",
    )

    suite = train_test_validation()

    result = suite.run(
        train_data,
        test_data,
        max_samples=5000,
    )

    result.save_as_html(
        "deepchecks_report.html",
        as_widget=False,
    )

    return result