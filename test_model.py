import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import test_loader
from model import FashionMNISTModel

checkpoint = torch.load("models/FashionMNIST.pth")
model = FashionMNISTModel()
model.load_state_dict(checkpoint['model_state_dict'])

test_loss = 0
acc = 0
loss_fn = nn.CrossEntropyLoss()

model.eval()
with torch.no_grad():
    for features, targets in tqdm(test_loader, desc="Test loader"):
        pred = model(features)
        loss = loss_fn(pred, targets)
        test_loss += loss.item()
        predicted_classes = pred.argmax(dim=1)
        acc += (predicted_classes == targets).sum().item() / len(targets)
    test_loss /= len(test_loader)
    acc /= len(test_loader)

print(f"Test Loss: {test_loss:.4f} | Accuracy: {acc * 100:.2f}%")