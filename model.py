import torch.nn as nn


class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, bias=False), # 26x26
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, bias=False), # 24x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 12x12
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, bias=False),  # 10x10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # 5x5
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(64*5*5, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        out = self.linear(x)
        return out
