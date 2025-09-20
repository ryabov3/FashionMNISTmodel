import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset import train_loader, val_loader
from earlystopping import Earlystopping
from model import FashionMNISTModel

model = FashionMNISTModel()
lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=2)
loss_fn = nn.CrossEntropyLoss()
earlystopping = Earlystopping(attemps=5)

save_path = "models/FashionMNIST.pth"
all_train_loss = []
all_val_loss = []
all_test_loss = []
all_acc = []

best_loss = float("inf")
EPOCH = 65

for num_epoch in range(1, EPOCH + 1):
    train_loss = 0

    model.train()
    for features, targets in tqdm(train_loader, desc=f"Epoch [{num_epoch}/{EPOCH}] | Train loader"):
        optimizer.zero_grad()
        pred = model(features)
        loss = loss_fn(pred, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)

    val_loss = 0
    acc = 0
    model.eval()
    with torch.no_grad():
        for features, targets in tqdm(val_loader, desc=f"Epoch [{num_epoch}/{EPOCH}] | Val loader"):
            pred = model(features)
            loss = loss_fn(pred, targets)
            val_loss += loss.item()
            predicted_classes = pred.argmax(dim=1)
            acc += (predicted_classes == targets).sum().item() / len(targets)
        val_loss /= len(val_loader)
        acc /= len(val_loader)
    
    logger.info(f"Epoch [{num_epoch}/{EPOCH}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {acc * 100:.2f}% | lr: {optimizer.param_groups[0]["lr"]}")

    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        save_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": num_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc         
        }
        torch.save(save_state, save_path)
        logger.info(f"Лучшая модель сохранена на {num_epoch} эпохе. Путь: {save_path}")
    
    if earlystopping(val_loss):
        logger.warning(f"Сработала ранняя остановка на {num_epoch} эпохе.")
        break
