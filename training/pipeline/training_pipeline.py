from pathlib import Path
import hashlib

from zenml import pipeline, step

from training.data_preparation import prep_data
from training.data_validation import validate_data
from training.train import train_model
from training.evaluate import eval_model
from training.model_promotion import promote_model


TRAINING_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = TRAINING_DIR / "Core_Dataset" / "raw"


def get_dataset_fingerprint() -> str:
    hasher = hashlib.sha256()

    for file_path in sorted(RAW_DATA_DIR.rglob("*")):

        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(
            RAW_DATA_DIR
        )

        hasher.update(
            str(relative_path).encode()
        )

        hasher.update(
            str(file_path.stat().st_size).encode()
        )

        hasher.update(
            str(file_path.stat().st_mtime_ns).encode()
        )

    return hasher.hexdigest()


@step(enable_cache=False)
def dataset_fingerprint_step() -> str:
    return get_dataset_fingerprint()


@step
def validation_step(
    dataset_fingerprint: str
) -> str:

    validate_data()

    return dataset_fingerprint


@step
def preparation_step(
    validated_fingerprint: str
) -> str:

    prep_data()

    return validated_fingerprint


@step(enable_cache=False)
def training_step(
    prepared_fingerprint: str
) -> str:

    return train_model()


@step(enable_cache=False)
def evaluation_step(
    model_path: str
) -> dict:

    return eval_model(
        model_path
    )


@step(enable_cache=False)
def promotion_step(
    model_path: str,
    metrics: dict,
) -> bool:

    return promote_model(
        model_path,
        metrics
    )


@pipeline
def training_pipeline() -> None:

    dataset_fingerprint = (
        dataset_fingerprint_step()
    )

    validated_fingerprint = (
        validation_step(
            dataset_fingerprint
        )
    )

    prepared_fingerprint = (
        preparation_step(
            validated_fingerprint
        )
    )

    model_path = training_step(
        prepared_fingerprint
    )

    metrics = evaluation_step(
        model_path
    )

    promotion_step(
        model_path,
        metrics
    )


if __name__ == "__main__":
    training_pipeline()