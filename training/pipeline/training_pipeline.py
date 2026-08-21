from pathlib import Path
import hashlib

from zenml import pipeline, step

from training.data_preparation import prep_data
from training.data_validation import validate_data
from training.train import train_model
from training.evaluate import eval_model
from training.model_promotion import promote_model
from training.deepchecks_validation import run_deepchecks


TRAINING_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PROJECT_ROOT = (
    TRAINING_DIR.parent
)

RAW_DATA_DIR = (
    TRAINING_DIR
    / "Core_Dataset"
    / "raw"
)

PROCESSED_DATA_DIR = (
    TRAINING_DIR
    / "Core_Dataset"
    / "processed_v1"
)

CI_DATA_DIR = (
    TRAINING_DIR
    / "ci_dataset"
    / "processed_v1"
)

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "best_model.pt"
)


def get_dataset_fingerprint() -> str:

    hasher = hashlib.sha256()

    for file_path in sorted(
        RAW_DATA_DIR.rglob("*")
    ):

        if not file_path.is_file():
            continue

        relative_path = (
            file_path.relative_to(
                RAW_DATA_DIR
            )
        )

        hasher.update(
            str(
                relative_path
            ).encode()
        )

        hasher.update(
            str(
                file_path.stat().st_size
            ).encode()
        )

        hasher.update(
            str(
                file_path.stat().st_mtime_ns
            ).encode()
        )

    return hasher.hexdigest()


@step(enable_cache=False)
def dataset_fingerprint_step() -> str:

    return get_dataset_fingerprint()


@step
def validation_step(
    dataset_fingerprint: str,
) -> str:

    validate_data()

    return dataset_fingerprint


@step
def preparation_step(
    validated_fingerprint: str,
) -> str:

    prep_data()

    return validated_fingerprint


@step(enable_cache=False)
def deepcheck_data_step(
    prepared_fingerprint: str,
) -> str:

    run_deepchecks()

    return prepared_fingerprint


# ---------------------------------
# NORMAL TRAINING STEP
# Requires your MLflow tracker.
# ---------------------------------

@step(
    enable_cache=False,
    experiment_tracker="mlflow_tracker",
)
def training_step(
    data_dir: str,
    resume: bool = False,
    epochs: int = 5,
) -> str:

    return train_model(
        resume=resume,
        epochs=epochs,
        data_dir=Path(data_dir),
        log_to_mlflow=True,
    )


# ---------------------------------
# CI TRAINING STEP
# No MLflow tracker required.
# ---------------------------------

@step(enable_cache=False)
def ci_training_step(
    data_dir: str,
) -> str:

    return train_model(
        resume=False,
        epochs=1,
        data_dir=Path(data_dir),
        log_to_mlflow=False,
    )


@step(
    enable_cache=False,
    experiment_tracker="mlflow_tracker",
)
def candidate_evaluation_step(
    model_path: str,
) -> dict:

    return eval_model(
        model_path,
        log_to_mlflow=True,
    )


@step(enable_cache=False)
def production_evaluation_step() -> dict:

    if not PRODUCTION_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Production model does not exist:\n"
            f"{PRODUCTION_MODEL_PATH}"
        )

    return eval_model(
        str(
            PRODUCTION_MODEL_PATH
        ),
        log_to_mlflow=False,
    )


@step(enable_cache=False)
def promotion_step(
    candidate_model_path: str,
    candidate_metrics: dict,
    production_metrics: dict,
) -> bool:

    return promote_model(
        candidate_model_path,
        candidate_metrics,
        production_metrics,
    )


@pipeline
def training_pipeline(
    resume: bool = False,
    epochs: int = 5,
    ci_mode: bool = False,
) -> None:

    # ---------------------------------
    # CI SMOKE TEST
    # ---------------------------------

    if ci_mode:

        print(
            "Running ZenML CI smoke test"
        )

        ci_training_step(
            data_dir=str(
                CI_DATA_DIR
            )
        )

        return

    # ---------------------------------
    # NORMAL TRAINING PIPELINE
    # ---------------------------------

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

    deepcheck_data_step(
        prepared_fingerprint
    )

    candidate_model_path = (
        training_step(
            data_dir=str(
                PROCESSED_DATA_DIR
            ),
            resume=resume,
            epochs=epochs,
        )
    )

    candidate_metrics = (
        candidate_evaluation_step(
            candidate_model_path
        )
    )

    production_metrics = (
        production_evaluation_step()
    )

    promotion_step(
        candidate_model_path,
        candidate_metrics,
        production_metrics,
    )


if __name__ == "__main__":
    training_pipeline()