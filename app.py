import os
import download_model

from flask import (
    Flask,
    render_template,
    request
)

from predict import predict_image

from model_metrics import (
    get_model_metrics,
    generate_graphs
)

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    "static/outputs",
    exist_ok=True
)

os.makedirs(
    "static/graphs",
    exist_ok=True
)

print("Evaluating model...")

metrics = get_model_metrics()

generate_graphs(metrics)

print("Metrics loaded successfully")


@app.route("/", methods=["GET", "POST"])
def index():

    original_image = None
    mask_image = None
    overlay_image = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                metrics=metrics
            )

        file = request.files["image"]

        if file and file.filename != "":

            upload_path = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(upload_path)

            (
                original_image,
                mask_image,
                overlay_image,
                _
            ) = predict_image(upload_path)

    return render_template(

        "index.html",

        original_image=original_image,

        mask_image=mask_image,

        overlay_image=overlay_image,

        metrics=metrics
    )


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )




    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )