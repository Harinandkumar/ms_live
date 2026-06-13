import os
import torch
import cv2
import numpy as np

from model import AttentionUNet

DEVICE = "cpu"

model = None


def get_model():

    global model

    if model is None:

        print("Loading model...")

        model = AttentionUNet().to(DEVICE)

        model.load_state_dict(
            torch.load(
                "saved_models/best_model.pth",
                map_location=DEVICE
            )
        )

        model.eval()

        print("Model loaded successfully")

    return model


def predict_image(image_path):

    model = get_model()

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    original = image.copy()

    image = cv2.resize(
        image,
        (256, 256)
    )

    original = cv2.resize(
        original,
        (256, 256)
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    image = torch.from_numpy(
        image
    ).float().to(DEVICE)

    with torch.no_grad():

        pred = model(image)

    pred = torch.sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()

    binary = (
        pred > 0.5
    ).astype(np.uint8)

    mask = (
        binary * 255
    ).astype(np.uint8)

    overlay = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    overlay[binary == 1] = [255, 0, 0]

    original_path = os.path.join(
        "static/outputs",
        "original.png"
    )

    mask_path = os.path.join(
        "static/outputs",
        "mask.png"
    )

    overlay_path = os.path.join(
        "static/outputs",
        "overlay.png"
    )

    cv2.imwrite(
        original_path,
        original
    )

    cv2.imwrite(
        mask_path,
        mask
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    return (
        original_path,
        mask_path,
        overlay_path,
        {}
    )
