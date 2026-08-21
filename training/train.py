from pathlib import Path

import mlflow
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from backend.src.model import ChickenCNN


TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent

DATA_DIR = (
    TRAINING_DIR
    / "Core_Dataset"
    / "processed_v1"
)

CANDIDATE_MODEL_DIR = (
    TRAINING_DIR
    / "models"
)

CANDIDATE_MODEL_FILE = (
    CANDIDATE_MODEL_DIR
    / "candidate_model.pt"
)

PRODUCTION_MODEL_FILE = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "best_model.pt"
)


# TRAINING SPECS

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
CHICKEN_WEIGHT_MULTIPLIER = 1.5
DATASET_VERSION = "processed_v1"


CANDIDATE_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def train_model(
    resume: bool = False,
    epochs: int = EPOCHS,
    data_dir: Path = DATA_DIR,
    log_to_mlflow: bool = True,
) -> str:

    device = torch.device("cpu")

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=val_transform,
    )

    print("Training data:", data_dir)
    print("Classes:", train_dataset.classes)
    print("Class mapping:", train_dataset.class_to_idx)

    class_counts = [
        0
    ] * len(train_dataset.classes)

    for _, label in train_dataset.samples:
        class_counts[label] += 1

    print(
        "Training class counts:",
        class_counts,
    )

    if len(class_counts) != 3:
        raise ValueError(
            "Expected exactly 3 classes."
        )

    if any(
        count == 0
        for count in class_counts
    ):
        raise ValueError(
            "Every class must contain at least "
            "one training image."
        )

    class_weights = [
        CHICKEN_WEIGHT_MULTIPLIER
        / class_counts[0],

        1 / class_counts[1],

        1 / class_counts[2],
    ]

    sample_weights = [
        class_weights[label]
        for _, label in train_dataset.samples
    ]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ChickenCNN(
        num_classes=3
    ).to(device)

    if resume:

        if not PRODUCTION_MODEL_FILE.exists():
            raise FileNotFoundError(
                "Cannot continue training because "
                "the production checkpoint does "
                "not exist:\n"
                f"{PRODUCTION_MODEL_FILE}"
            )

        print("\nCONTINUING TRAINING")

        print(
            "Loading production checkpoint from:\n"
            f"{PRODUCTION_MODEL_FILE}"
        )

        state_dict = torch.load(
            PRODUCTION_MODEL_FILE,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            state_dict
        )

    else:

        print(
            "\nTRAINING FROM SCRATCH"
        )

        print(
            "No existing model weights "
            "will be loaded."
        )

    # -------------------------
    # MLFLOW PARAMETERS
    # -------------------------

    if log_to_mlflow:

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE,
        )

        mlflow.log_param(
            "epochs",
            epochs,
        )

        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE,
        )

        mlflow.log_param(
            "training_mode",
            "continue"
            if resume
            else "scratch",
        )

        mlflow.log_param(
            "optimizer",
            "Adam",
        )

        mlflow.log_param(
            "loss_function",
            "CrossEntropyLoss",
        )

        mlflow.log_param(
            "image_size",
            "128x128",
        )

        mlflow.log_param(
            "num_classes",
            3,
        )

        mlflow.log_param(
            "chicken_weight_multiplier",
            CHICKEN_WEIGHT_MULTIPLIER,
        )

        mlflow.log_param(
            "dataset_version",
            DATASET_VERSION,
        )

        mlflow.log_param(
            "data_dir",
            str(data_dir),
        )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = -1.0

    for epoch in range(epochs):

        # -------------------------
        # TRAINING
        # -------------------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            running_loss
            / total
        )

        train_accuracy = (
            correct
            / total
        )

        # -------------------------
        # VALIDATION
        # -------------------------

        model.eval()

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

                val_loss_total += (
                    loss.item()
                    * images.size(0)
                )

                _, predictions = torch.max(
                    outputs,
                    1,
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = (
            val_loss_total
            / val_total
        )

        val_accuracy = (
            val_correct
            / val_total
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # -------------------------
        # MLFLOW METRICS
        # -------------------------

        if log_to_mlflow:

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "train_accuracy",
                train_accuracy,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch,
            )

        # -------------------------
        # SAVE BEST MODEL
        # -------------------------

        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            torch.save(
                model.state_dict(),
                CANDIDATE_MODEL_FILE,
            )

            print(
                "Saved new best candidate model "
                "with validation accuracy "
                f"{best_val_accuracy:.4f}"
            )

    if log_to_mlflow:

        mlflow.log_metric(
            "best_val_accuracy",
            best_val_accuracy,
        )

        mlflow.log_param(
            "candidate_model_path",
            str(
                CANDIDATE_MODEL_FILE
            ),
        )

    print(
        "\nTraining complete."
    )

    print(
        "Best validation accuracy:",
        best_val_accuracy,
    )

    print(
        "Candidate model saved to:",
        CANDIDATE_MODEL_FILE,
    )

    return str(
        CANDIDATE_MODEL_FILE
    )


if __name__ == "__main__":
    train_model()