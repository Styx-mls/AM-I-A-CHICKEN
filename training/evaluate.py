from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backend.src.model import ChickenCNN


TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR / "Core_Dataset" / "processed_v1"

BATCH_SIZE = 32


def eval_model(
    model_path: str,
    log_to_mlflow: bool = True,
) -> dict:

    device = torch.device("cpu")

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model does not exist:\n{model_path}"
        )

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ChickenCNN(
        num_classes=3
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    correct = 0
    total = 0

    num_classes = len(
        test_dataset.classes
    )

    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.int64
    )

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            for true_label, predicted_label in zip(
                labels.cpu(),
                predictions.cpu()
            ):

                confusion_matrix[
                    true_label,
                    predicted_label
                ] += 1

    accuracy = correct / total

    print("\n--- TEST RESULTS ---")

    print(
        f"Model: {model_path}"
    )

    print(
        f"Test accuracy: {accuracy:.4f}"
    )

    print("\nClasses:")
    print(
        test_dataset.class_to_idx
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix
    )

    print("\n--- PER-CLASS ACCURACY ---")

    per_class_accuracy = {}

    for class_index, class_name in enumerate(
        test_dataset.classes
    ):

        class_correct = confusion_matrix[
            class_index,
            class_index
        ].item()

        class_total = confusion_matrix[
            class_index
        ].sum().item()

        if class_total > 0:
            class_accuracy = (
                class_correct / class_total
            )
        else:
            class_accuracy = 0.0

        per_class_accuracy[
            class_name
        ] = class_accuracy

        print(
            f"{class_name}: "
            f"{class_accuracy:.4f}"
        )

    # Only log when this evaluation is running
    # inside the ZenML MLflow-tracked evaluation step.

    if log_to_mlflow:

        mlflow.log_param(
            "evaluated_model_path",
            str(model_path)
        )

        mlflow.log_metric(
            "test_accuracy",
            accuracy
        )

        mlflow.log_metric(
            "test_chicken_accuracy",
            per_class_accuracy["chicken"]
        )

        mlflow.log_metric(
            "test_human_accuracy",
            per_class_accuracy["human"]
        )

        mlflow.log_metric(
            "test_other_accuracy",
            per_class_accuracy["other"]
        )

        mlflow.log_dict(
            {
                "classes": test_dataset.classes,
                "class_to_idx": test_dataset.class_to_idx,
                "confusion_matrix": confusion_matrix.tolist(),
            },
            "confusion_matrix.json",
        )

    return {
        "model_path": str(model_path),
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
    }


if __name__ == "__main__":
    raise RuntimeError(
        "evaluate.py requires a model path. "
        "Run it through the ZenML pipeline."
    )