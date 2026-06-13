import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    confusion_matrix
)

from dataset import MSDataset
from model import AttentionUNet


# =========================================================
# DEVICE
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)

# =========================================================
# OUTPUT FOLDER
# =========================================================

os.makedirs(
    "outputs/graphs",
    exist_ok=True
)

# =========================================================
# DATASET
# =========================================================

test_dataset = MSDataset(
    "processed_dataset/val/images",
    "processed_dataset/val/masks"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)

# =========================================================
# LOAD MODEL
# =========================================================

model = AttentionUNet().to(DEVICE)

model.load_state_dict(

    torch.load(
        "saved_models/best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

# =========================================================
# VARIABLES
# =========================================================

total_dice = 0
total_acc = 0
total_f1 = 0
total_precision = 0
total_recall = 0
total_iou = 0

all_preds = []
all_masks = []

# =========================================================
# DICE HISTORY (YOUR TRAINING VALUES)
# =========================================================

dice_scores = [

    0.4202,
    0.5818,
    0.5769,
    0.6002,
    0.6109,
    0.5901,
    0.5772,
    0.6077,
    0.5975,
    0.6026,
    0.6228,
    0.5946,
    0.6238,
    0.6238,
    0.5780,
    0.5969,
    0.6204,
    0.6278,
    0.6050,
    0.6250,
    0.6118,
    0.6240,
    0.5947,
    0.6306,
    0.6137,
    0.6327
]

# =========================================================
# LOSS HISTORY
# =========================================================

train_loss = [

    0.5615,
    0.4343,
    0.3312,
    0.2711,
    0.2329,
    0.2232,
    0.2128,
    0.2031,
    0.2034,
    0.1930,
    0.1861,
    0.1888,
    0.1818,
    0.1770,
    0.1800,
    0.1705,
    0.1703,
    0.1613,
    0.1619,
    0.1590,
    0.1551,
    0.1517,
    0.1523,
    0.1446,
    0.1432,
    0.1432
]

val_loss = [

    0.4982,
    0.3731,
    0.2976,
    0.2547,
    0.2403,
    0.2433,
    0.2469,
    0.2302,
    0.2357,
    0.2306,
    0.2228,
    0.2357,
    0.2194,
    0.2207,
    0.2445,
    0.2342,
    0.2214,
    0.2200,
    0.2309,
    0.2192,
    0.2282,
    0.2220,
    0.2377,
    0.2180,
    0.2268,
    0.2159
]

# =========================================================
# METRICS FUNCTION
# =========================================================

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

# =========================================================
# EVALUATION LOOP
# =========================================================

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)

        outputs = model(images)

        preds = torch.sigmoid(outputs)

        preds = (preds > 0.5).float()

        all_preds.extend(
            preds.view(-1).cpu().numpy()
        )

        all_masks.extend(
            masks.view(-1).cpu().numpy()
        )

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

# =========================================================
# FINAL METRICS
# =========================================================

avg_dice = total_dice / len(test_loader)

avg_acc = total_acc / len(test_loader)

avg_f1 = total_f1 / len(test_loader)

avg_precision = total_precision / len(test_loader)

avg_recall = total_recall / len(test_loader)

avg_iou = total_iou / len(test_loader)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n========== FINAL RESULTS ==========\n")

print(f"Dice Score : {avg_dice:.4f}")

print(f"Accuracy   : {avg_acc:.4f}")

print(f"F1 Score   : {avg_f1:.4f}")

print(f"Precision  : {avg_precision:.4f}")

print(f"Recall     : {avg_recall:.4f}")

print(f"IoU Score  : {avg_iou:.4f}")

print("\n===================================")

# =========================================================
# PROFESSIONAL LOSS GRAPH
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(
    range(1, len(train_loss)+1),
    train_loss,
    linewidth=3,
    marker='o',
    label="Training Loss"
)

plt.plot(
    range(1, len(val_loss)+1),
    val_loss,
    linewidth=3,
    marker='s',
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel(
    "Epochs",
    fontsize=13
)

plt.ylabel(
    "Loss",
    fontsize=13
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/graphs/professional_loss_graph.png",
    dpi=300
)

plt.close()

# =========================================================
# PROFESSIONAL DICE GRAPH
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(

    range(1, len(dice_scores)+1),

    dice_scores,

    linewidth=3,

    marker='o',

    label="Dice Score"
)

plt.title(
    "Dice Score vs Epochs",
    fontsize=16,
    fontweight='bold'
)

plt.xlabel(
    "Epochs",
    fontsize=13
)

plt.ylabel(
    "Dice Score",
    fontsize=13
)

plt.ylim(0,1)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/graphs/professional_dice_graph.png",
    dpi=300
)

plt.close()

# =========================================================
# PROFESSIONAL BAR GRAPH
# =========================================================

metrics_names = [

    "Dice",
    "Accuracy",
    "F1",
    "Precision",
    "Recall",
    "IoU"
]

metrics_values = [

    avg_dice,
    avg_acc,
    avg_f1,
    avg_precision,
    avg_recall,
    avg_iou
]

plt.figure(figsize=(11,6))

bars = plt.bar(
    metrics_names,
    metrics_values
)

plt.ylim(0,1)

plt.title(
    "Final Evaluation Metrics",
    fontsize=16,
    fontweight='bold'
)

plt.ylabel(
    "Score",
    fontsize=13
)

plt.grid(axis='y')

for bar in bars:

    yval = bar.get_height()

    plt.text(

        bar.get_x() + 0.15,

        yval + 0.01,

        f"{yval:.3f}",

        fontsize=11,

        fontweight='bold'
    )

plt.tight_layout()

plt.savefig(
    "outputs/graphs/professional_metrics_graph.png",
    dpi=300
)

plt.close()

# =========================================================
# PROFESSIONAL LINE GRAPH
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(
    metrics_names,
    metrics_values,
    linewidth=3,
    marker='o'
)

plt.ylim(0,1)

plt.title(
    "All Metrics Comparison",
    fontsize=16,
    fontweight='bold'
)

plt.ylabel(
    "Score",
    fontsize=13
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/graphs/professional_all_parameters_graph.png",
    dpi=300
)

plt.close()

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_masks,
    all_preds
)

plt.figure(figsize=(7,7))

plt.imshow(cm)

plt.title(
    "Confusion Matrix",
    fontsize=16,
    fontweight='bold'
)

plt.colorbar()

classes = ["Negative", "Positive"]

tick_marks = np.arange(len(classes))

plt.xticks(
    tick_marks,
    classes,
    fontsize=12
)

plt.yticks(
    tick_marks,
    classes,
    fontsize=12
)

plt.xlabel(
    "Predicted",
    fontsize=13
)

plt.ylabel(
    "Actual",
    fontsize=13
)

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            format(cm[i, j], 'd'),
            ha="center",
            va="center",
            fontsize=12,
            fontweight='bold'
        )

plt.tight_layout()

plt.savefig(
    "outputs/graphs/professional_confusion_matrix.png",
    dpi=300
)

plt.close()

# =========================================================
# INDIVIDUAL PROFESSIONAL METRICS
# =========================================================

graphs = [

    ("dice_bar_graph.png", "Dice Score", avg_dice),

    ("accuracy_bar_graph.png", "Accuracy", avg_acc),

    ("f1_bar_graph.png", "F1 Score", avg_f1),

    ("precision_bar_graph.png", "Precision", avg_precision),

    ("recall_bar_graph.png", "Recall", avg_recall),

    ("iou_bar_graph.png", "IoU Score", avg_iou)
]

for filename, title, value in graphs:

    plt.figure(figsize=(6,5))

    bars = plt.bar(
        [title],
        [value]
    )

    plt.ylim(0,1)

    plt.title(
        title,
        fontsize=15,
        fontweight='bold'
    )

    plt.ylabel(
        "Score",
        fontsize=12
    )

    for bar in bars:

        yval = bar.get_height()

        plt.text(

            bar.get_x() + 0.05,

            yval + 0.01,

            f"{yval:.3f}",

            fontsize=11,

            fontweight='bold'
        )

    plt.tight_layout()

    plt.savefig(
        f"outputs/graphs/{filename}",
        dpi=300
    )

    plt.close()

# =========================================================
# FINISHED
# =========================================================

print("\nALL PROFESSIONAL GRAPHS SAVED SUCCESSFULLY")

print("\nSaved in:")

print("outputs/graphs/")