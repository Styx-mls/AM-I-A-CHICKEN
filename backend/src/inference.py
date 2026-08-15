from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import ChickenCNN


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pt"

CLASS_NAMES = [
    "chicken",
    "human",
    "other"
]

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

model = ChickenCNN(num_classes=3).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

def model_ready():
    return model is not None


def predict_image(image):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]

    return {
        "prediction": predicted_class,
        "confidence": confidence.item(),
        "probabilities": {
            CLASS_NAMES[i]: probabilities[0][i].item()
            for i in range(len(CLASS_NAMES))
        }
    }


if __name__ == "__main__":

    test_image = "test_image.jpg"

    result = predict_image(test_image)

    print(result)