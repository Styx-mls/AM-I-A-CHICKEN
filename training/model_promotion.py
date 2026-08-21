from pathlib import Path
import shutil


TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "best_model.pt"
)


def promote_model(
    candidate_model_path: str,
    candidate_metrics: dict,
    production_metrics: dict,
) -> bool:

    candidate_model_path = Path(
        candidate_model_path
    )

    if not candidate_model_path.exists():
        raise FileNotFoundError(
            f"Candidate model does not exist:\n"
            f"{candidate_model_path}"
        )

    candidate_accuracy = (
        candidate_metrics["accuracy"]
    )

    candidate_chicken_accuracy = (
        candidate_metrics[
            "per_class_accuracy"
        ]["chicken"]
    )

    production_accuracy = (
        production_metrics["accuracy"]
    )

    production_chicken_accuracy = (
        production_metrics[
            "per_class_accuracy"
        ]["chicken"]
    )

    print("\n--- MODEL PROMOTION ---")

    print(
        f"Candidate overall accuracy: "
        f"{candidate_accuracy:.4f}"
    )

    print(
        f"Candidate chicken accuracy: "
        f"{candidate_chicken_accuracy:.4f}"
    )

    print(
        f"Production overall accuracy: "
        f"{production_accuracy:.4f}"
    )

    print(
        f"Production chicken accuracy: "
        f"{production_chicken_accuracy:.4f}"
    )

    overall_better = (
        candidate_accuracy
        > production_accuracy
    )

    chicken_better = (
        candidate_chicken_accuracy
        > production_chicken_accuracy
    )

    if overall_better and chicken_better:

        PRODUCTION_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            candidate_model_path,
            PRODUCTION_MODEL_PATH
        )

        print(
            "\nCandidate improved BOTH overall accuracy "
            "and chicken accuracy."
        )

        print(
            "Candidate promoted to production."
        )

        return True

    print(
        "\nCandidate was NOT promoted."
    )

    if not overall_better:
        print(
            "Overall accuracy did not improve."
        )

    if not chicken_better:
        print(
            "Chicken accuracy did not improve."
        )

    print(
        "Current production model remains unchanged."
    )

    return False