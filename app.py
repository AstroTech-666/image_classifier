from flask import Flask, request, jsonify, render_template
from torchvision import transforms, models
import torch
from PIL import Image
import json
from torch import nn

app = Flask(__name__, template_folder="templates")

# Load model (use vgg16_bn for batch-normalized version)
model = models.vgg16_bn(pretrained=False)

# Define classifier to match the saved checkpoint
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 102),
    nn.LogSoftmax(dim=1)
)

checkpoint_path = "vgg16_bn_checkpoint.pth"
cat_to_name_path = "cat_to_name.json"

try:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
except FileNotFoundError:
    print(f"Error: Checkpoint file '{checkpoint_path}' not found!")
    exit(1)

state_dict = checkpoint["state_dict"]
new_state_dict = {}

for key in state_dict:
    new_key = key.replace("classifier.fc1", "classifier.0") \
                 .replace("classifier.fc2", "classifier.3") \
                 .replace("classifier.fc3", "classifier.6")
    new_state_dict[new_key] = state_dict[key]

model.load_state_dict(new_state_dict)
model.eval()

try:
    with open(cat_to_name_path, "r") as f:
        cat_to_name = json.load(f)
except FileNotFoundError:
    print(f"Error: Category file '{cat_to_name_path}' not found!")
    exit(1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted_class = output.max(1)

        predicted_class_str = str(predicted_class.item())

        if predicted_class_str not in cat_to_name:
            return jsonify({"error": "Prediction not found in category names"}), 500

        flower_name = cat_to_name[predicted_class_str]
        return jsonify({"flower": flower_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)