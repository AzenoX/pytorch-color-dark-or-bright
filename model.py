"""
Use this file to use the trained model
"""

import os
import sys
import torch
from torch import nn
from utils import get_model_path, get_truth_table

# Model Path
MODEL_SAVE_PATH = get_model_path()

TRUTH_TABLE = get_truth_table()

if __name__ == '__main__':
    if os.path.isfile(MODEL_SAVE_PATH):  # If the model already exists, then load it's dict
        model = nn.Sequential(
            nn.Linear(in_features=3, out_features=5),
            nn.Linear(in_features=5, out_features=1)
        )
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

        if len(sys.argv) == 4:
            red = int(sys.argv[1])
            green = int(sys.argv[2])
            blue = int(sys.argv[3])

            X = torch.tensor([
                [red, green, blue],
            ], dtype=torch.float32)
            # Normalize X_real
            X = X / 255
            with torch.inference_mode():
                model.eval()
                y_logits = model(X)
                y_pred = torch.round(torch.sigmoid(y_logits))

                # Finally print the result
                print("Determining if a color is bright or dark is subjective."
                      "The model is based on the luminance equation "
                      "(0.299 * R + 0.587 * G + 0.114 * B), with a threshold of 128.")
                print(f"The color {red}, {green}, {blue} is {TRUTH_TABLE[y_pred.item()]}")
    else:
        print('You need to have a model first (run: `python main.py`)')
