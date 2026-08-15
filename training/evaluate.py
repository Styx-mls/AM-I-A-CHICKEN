from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ChickenCNN


DATA_DIR = Path("Core_Dataset/processed_v1/test")
MODEL_PATH = Path("models/best_model.pt")

BATCH_SIZE = 32

device = torch.device("cpu")


test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


test_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = ChickenCNN(num_classes=3).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


correct = 0
total = 0

num_classes = len(test_dataset.classes)

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

        _, predictions = torch.max(outputs, 1)

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
print(f"Test accuracy: {accuracy:.4f}")

print("\nClasses:")
print(test_dataset.class_to_idx)

print("\nConfusion matrix:")
print(confusion_matrix)


print("\n--- PER-CLASS ACCURACY ---")

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

    class_accuracy = (
        class_correct / class_total
    )

    print(
        f"{class_name}: "
        f"{class_accuracy:.4f}"
    )