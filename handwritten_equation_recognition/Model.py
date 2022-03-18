import torch.nn as nn




model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),
    nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),
    nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),
    nn.Flatten(),
    nn.Linear(64*5*5, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 12)

)
print("model loaded")
