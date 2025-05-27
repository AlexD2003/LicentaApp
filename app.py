import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b1
from torch import nn
import cv2
from flask import Flask, render_template, request
from utils.image_loader import save_and_convert
from utils.preprocessing import full_preprocess
import os

app = Flask(__name__)
# Load model once
MODEL_PATH = "model/cbis_efficientnet_b1.pth"
THRESHOLD = 0.75

model = efficientnet_b1()
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

app.config["UPLOAD_FOLDER"] = "static/uploads"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", original=None, processed=None)

@app.route("/upload", methods=["POST"])
def upload():
    original = None
    processed = None

    image_file = request.files.get("image")
    if image_file:
        image_path = save_and_convert(image_file, save_dir=app.config["UPLOAD_FOLDER"])
        original = "/" + image_path

    return render_template("index.html", original=original, processed=None)

@app.route("/preprocess", methods=["POST"])
def preprocess():
    original = request.form.get("original")
    processed = None
    prediction = None

    if original:
        local_path = original.lstrip("/")
        processed_path = os.path.splitext(local_path)[0] + "_processed.png"
        full_preprocess(local_path, processed_path)
        processed = "/" + processed_path

        # Predict
        prediction = predict_from_image(processed_path)

    return render_template("index.html", original=original, processed=processed, prediction=prediction)

def predict_from_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (240, 240))
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    # 🔥 FIX: apply the same normalization as during training
    tensor = transforms.Normalize((0.5,), (0.5,))(tensor)

    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)[0][1].item()
        label = "Malignant" if prob >= THRESHOLD else "Benign"
        return f"{label} (Prob: {prob:.2f})"


if __name__ == "__main__":
    app.run(debug=True)
