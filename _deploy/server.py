"""
Server file
"""

from flask import Flask, request, jsonify
import torch
from torch import nn

app = Flask(__name__)

MODEL_SAVE_PATH = "models/model_colors.pt"
TRUTH_TABLE = {0: 'Bright', 1: 'Dark'}

model = nn.Sequential(
    nn.Linear(in_features=3, out_features=5),
    nn.Linear(in_features=5, out_features=1)
)

# Load the model
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the color
    """
    data = request.json
    red = data.get('red', 0)
    green = data.get('green', 0)
    blue = data.get('blue', 0)

    x = torch.tensor([[red, green, blue]], dtype=torch.float32)
    x = x / 255

    with torch.inference_mode():
        y_logits = model(x)
        y_pred = torch.round(torch.sigmoid(y_logits))

    return jsonify({
        "color": f"{red}, {green}, {blue}",
        "prediction": TRUTH_TABLE[y_pred.item()]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
