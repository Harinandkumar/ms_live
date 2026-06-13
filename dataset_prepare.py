import os
import cv2
import random
import albumentations as A

from tqdm import tqdm
from sklearn.model_selection import train_test_split


MRI_ROOT = "bmri_dataet/BMRI_Resized_MRI"
MASK_ROOT = "bmri_dataet/BMRI_Resized_Lesion"

OUTPUT = "processed_dataset"

all_pairs = []

print("Collecting dataset...")

patients = os.listdir(MRI_ROOT)

for patient in patients:

    mri_patient_path = os.path.join(
        MRI_ROOT,
        patient,
        "image"
    )

    mask_patient_path = os.path.join(
        MASK_ROOT,
        patient,
        "lesion"
    )

    if not os.path.exists(mri_patient_path):
        continue

    modalities = os.listdir(mri_patient_path)

    for modality in modalities:

        # ONLY FLAIR
        if "Flair" not in modality:
            continue

        mri_modality_path = os.path.join(
            mri_patient_path,
            modality
        )

        mask_modality = modality.replace(
            "Flair_slice",
            "LesionSeg-Flair_slice"
        )

        mask_modality_path = os.path.join(
            mask_patient_path,
            mask_modality
        )

        if not os.path.exists(mask_modality_path):
            continue

        mri_images = sorted(
            os.listdir(mri_modality_path)
        )

        mask_images = sorted(
            os.listdir(mask_modality_path)
        )

        for img_name, mask_name in zip(
            mri_images,
            mask_images
        ):

            img_path = os.path.join(
                mri_modality_path,
                img_name
            )

            mask_path = os.path.join(
                mask_modality_path,
                mask_name
            )

            all_pairs.append(
                (img_path, mask_path)
            )

print("TOTAL PAIRS:", len(all_pairs))

random.shuffle(all_pairs)

train_pairs, temp_pairs = train_test_split(
    all_pairs,
    test_size=0.3,
    random_state=42
)

val_pairs, test_pairs = train_test_split(
    temp_pairs,
    test_size=0.5,
    random_state=42
)

splits = {
    "train": train_pairs,
    "val": val_pairs,
    "test": test_pairs
}

# LIGHT AUGMENTATION ONLY
augment = A.Compose([

    A.HorizontalFlip(p=0.5),

    A.Rotate(limit=10, p=0.3)

])

for split_name, pairs in splits.items():

    image_out = os.path.join(
        OUTPUT,
        split_name,
        "images"
    )

    mask_out = os.path.join(
        OUTPUT,
        split_name,
        "masks"
    )

    os.makedirs(
        image_out,
        exist_ok=True
    )

    os.makedirs(
        mask_out,
        exist_ok=True
    )

    counter = 0

    print(f"\nProcessing {split_name}...")

    for img_path, mask_path in tqdm(pairs):

        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(
            image,
            (256, 256)
        )

        mask = cv2.resize(
            mask,
            (256, 256)
        )

        # SAVE ORIGINAL
        cv2.imwrite(
            os.path.join(
                image_out,
                f"{counter}.png"
            ),
            image
        )

        cv2.imwrite(
            os.path.join(
                mask_out,
                f"{counter}.png"
            ),
            mask
        )

        counter += 1

        # AUGMENT ONLY TRAIN
        if split_name == "train":

            for _ in range(4):

                augmented = augment(
                    image=image,
                    mask=mask
                )

                aug_img = augmented["image"]

                aug_mask = augmented["mask"]

                cv2.imwrite(
                    os.path.join(
                        image_out,
                        f"{counter}.png"
                    ),
                    aug_img
                )

                cv2.imwrite(
                    os.path.join(
                        mask_out,
                        f"{counter}.png"
                    ),
                    aug_mask
                )

                counter += 1

print("\nDATASET READY")