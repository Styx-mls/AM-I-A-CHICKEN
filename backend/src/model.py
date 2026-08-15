import torch
import torch.nn as nn

class ChickenCNN(nn.Module):

    def __init__(self, num_classes = 3):
        
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 32,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    model = ChickenCNN()

    test_input = torch.randn(4, 3, 128, 128)
    output = model(test_input)

    print(model)
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)