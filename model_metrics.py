import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    jaccard_score
)

from dataset import MSDataset
from model import AttentionUNet


DEVICE = "cpu"


def get_model_metrics():

    dataset = MSDataset(
        "processed_dataset/val/images",
        "processed_dataset/val/masks"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    model = AttentionUNet().to(DEVICE)

    model.load_state_dict(
        torch.load(
            "saved_models/best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    total_dice = 0
    total_acc = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_iou = 0

    total_batches = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            preds = torch.sigmoid(outputs)

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

            total_dice += dice.item()

            total_acc += acc.item()

            total_f1 += f1

            total_precision += precision

            total_recall += recall

            total_iou += iou

            total_batches += 1

    metrics = {

        "dice": round(
            total_dice / total_batches,
            4
        ),

        "accuracy": round(
            total_acc / total_batches,
            4
        ),

        "f1": round(
            total_f1 / total_batches,
            4
        ),

        "precision": round(
            total_precision / total_batches,
            4
        ),

        "recall": round(
            total_recall / total_batches,
            4
        ),

        "iou": round(
            total_iou / total_batches,
            4
        )
    }

    return metrics


def generate_graphs(metrics):

    names = [
        "Dice",
        "Accuracy",
        "F1",
        "Precision",
        "Recall",
        "IoU"
    ]

    values = [
        metrics["dice"],
        metrics["accuracy"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        metrics["iou"]
    ]

    # BAR GRAPH

    plt.figure(figsize=(10,6))

    bars = plt.bar(
        names,
        values
    )

    plt.ylim(0,1)

    plt.title(
        "Model Evaluation Metrics"
    )

    plt.ylabel(
        "Score"
    )

    for bar in bars:

        yval = bar.get_height()

        plt.text(
            bar.get_x() + 0.15,
            yval + 0.01,
            f"{yval:.3f}"
        )

    plt.savefig(
        "static/graphs/metrics_bar_graph.png"
    )

    plt.close()

    # LINE GRAPH

    plt.figure(figsize=(10,6))

    plt.plot(
        names,
        values,
        marker='o',
        linewidth=3
    )

    plt.ylim(0,1)

    plt.title(
        "Model Metrics Line Graph"
    )

    plt.ylabel(
        "Score"
    )

    plt.grid(True)

    plt.savefig(
        "static/graphs/metrics_line_graph.png"
    )

    plt.close()

    # INDIVIDUAL GRAPHS

    graph_data = [

        ("dice_graph.png", "Dice Score", metrics["dice"]),

        ("accuracy_graph.png", "Accuracy", metrics["accuracy"]),

        ("f1_graph.png", "F1 Score", metrics["f1"]),

        ("precision_graph.png", "Precision", metrics["precision"]),

        ("recall_graph.png", "Recall", metrics["recall"]),

        ("iou_graph.png", "IoU Score", metrics["iou"])
    ]

    for filename, title, value in graph_data:

        plt.figure(figsize=(6,5))

        bars = plt.bar(
            [title],
            [value]
        )

        plt.ylim(0,1)

        plt.title(title)

        plt.ylabel("Score")

        for bar in bars:

            yval = bar.get_height()

            plt.text(
                bar.get_x() + 0.05,
                yval + 0.01,
                f"{yval:.3f}"
            )

        plt.savefig(
            f"static/graphs/{filename}"
        )

        plt.close()