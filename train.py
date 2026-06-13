import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    jaccard_score
)

from dataset import MSDataset
from model import AttentionUNet
from losses import HybridLoss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)

os.makedirs(
    "saved_models",
    exist_ok=True
)

os.makedirs(
    "outputs/graphs",
    exist_ok=True
)

train_dataset = MSDataset(
    "processed_dataset/train/images",
    "processed_dataset/train/masks"
)

val_dataset = MSDataset(
    "processed_dataset/val/images",
    "processed_dataset/val/masks"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4
)

model = AttentionUNet().to(DEVICE)

criterion = HybridLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=5
)

scaler = torch.amp.GradScaler("cuda")

best_dice = 0

patience = 30
counter = 0

EPOCHS = 50

train_losses = []
val_losses = []
dice_scores = []


def metrics(preds, masks):

    preds = torch.sigmoid(preds)

    preds = (preds > 0.5).float()

    preds_flat = preds.view(-1).cpu().numpy()
    masks_flat = masks.view(-1).cpu().numpy()

    f1 = f1_score(
        masks_flat,
        preds_flat,
        average='binary',
        zero_division=1
    )

    precision = precision_score(
        masks_flat,
        preds_flat,
        zero_division=1
    )

    recall = recall_score(
        masks_flat,
        preds_flat,
        zero_division=1
    )

    iou = jaccard_score(
        masks_flat,
        preds_flat,
        zero_division=1
    )

    intersection = (
        preds * masks
    ).sum()

    dice = (
        2 * intersection
    ) / (
        preds.sum() +
        masks.sum() +
        1e-8
    )

    acc = (
        preds == masks
    ).float().mean()

    return (
        dice.item(),
        acc.item(),
        f1,
        precision,
        recall,
        iou
    )


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    loop = tqdm(train_loader)

    for images, masks in loop:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

        loop.set_description(
            f"Epoch [{epoch+1}/{EPOCHS}]"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    model.eval()

    val_loss = 0

    total_dice = 0
    total_acc = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_iou = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            val_loss += loss.item()

            (
                dice,
                acc,
                f1,
                precision,
                recall,
                iou
            ) = metrics(outputs, masks)

            total_dice += dice
            total_acc += acc
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            total_iou += iou

    avg_train_loss = train_loss / len(train_loader)

    avg_val_loss = val_loss / len(val_loader)

    avg_dice = total_dice / len(val_loader)

    avg_acc = total_acc / len(val_loader)

    avg_f1 = total_f1 / len(val_loader)

    avg_precision = total_precision / len(val_loader)

    avg_recall = total_recall / len(val_loader)

    avg_iou = total_iou / len(val_loader)

    train_losses.append(avg_train_loss)

    val_losses.append(avg_val_loss)

    dice_scores.append(avg_dice)

    scheduler.step(avg_dice)

    print("\n====================")

    print(f"Epoch: {epoch+1}")

    print(f"Train Loss: {avg_train_loss:.4f}")

    print(f"Val Loss: {avg_val_loss:.4f}")

    print(f"Dice: {avg_dice:.4f}")

    print(f"Accuracy: {avg_acc:.4f}")

    print(f"F1: {avg_f1:.4f}")

    print(f"Precision: {avg_precision:.4f}")

    print(f"Recall: {avg_recall:.4f}")

    print(f"IoU: {avg_iou:.4f}")

    print("====================\n")

    if avg_dice > best_dice:

        best_dice = avg_dice

        torch.save(
            model.state_dict(),
            "saved_models/best_model.pth"
        )

        print("BEST MODEL SAVED")

        counter = 0

    else:

        counter += 1

    if counter >= patience:

        print("EARLY STOPPING")

        break


plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Val Loss"
)

plt.legend()

plt.savefig(
    "outputs/graphs/loss_graph.png"
)

plt.clf()

plt.plot(
    dice_scores,
    label="Dice"
)

plt.legend()

plt.savefig(
    "outputs/graphs/dice_graph.png"
)

print("TRAINING COMPLETE")