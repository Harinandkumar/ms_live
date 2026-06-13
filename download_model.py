import gdown
import os

os.makedirs(
    "saved_models",
    exist_ok=True
)

url = "https://drive.google.com/uc?id=1gAabgK5rkXgA3am9puolIoLraQbUh356"

output = "saved_models/best_model.pth"

if not os.path.exists(output):

    print("Downloading model...")

    gdown.download(
        url,
        output,
        quiet=False
    )

    print("Model downloaded")

else:

    print("Model already exists")