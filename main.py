import sys
import os
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import accuracy_fn, visualize_decision_boundary_3d
from pathlib import Path


# Model Path
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "model_colors.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

TRUTH_TABLE = {
    0: 'Bright',
    1: 'Dark'
}

model = nn.Sequential(
    nn.Linear(in_features=3, out_features=5),
    nn.Linear(in_features=5, out_features=1)
)
if os.path.isfile(MODEL_SAVE_PATH):  # If the model already exists, then load it's dict
    model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
else:  # Else, train the model and save it's state dict
    file_path = "colors.xlsx"
    df = pd.read_excel(file_path, engine='openpyxl')
    # Convert the first column to 'y' numpy array of shape (z, 1)
    y = df.iloc[:, 0].values.reshape(-1, 1)
    # Convert the rest of the columns to 'X' numpy array of shape (z, t)
    X = df.iloc[:, 1:].values
    # Convert to tensors
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)
    # Normalize X
    X = X / 255
    # Create datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Define hyperparameters
    EPOCHS = 3000
    LEARNING_RATE = 0.05

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    loss_metrics = []
    # Run train
    for epoch in range(EPOCHS):
        model.train()
        # Forward pass
        y_logits = model(X_train)
        y_pred = torch.round(torch.sigmoid(y_logits))
        # Loss
        loss = loss_fn(y_logits, y_train)
        loss_metrics.append(loss.item())
        # Zero optimize
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Optimize
        optimizer.step()
        # Get Accuracy
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # Perform test pass
        with torch.inference_mode():
            model.eval()
            # Forward pass
            y_logits_test = model(X_test)
            y_pred_test = torch.round(torch.sigmoid(y_logits_test))
            # Loss
            loss_test = loss_fn(y_logits_test, y_test)
            # Get Accuracy
            acc_test = accuracy_fn(y_true=y_test, y_pred=y_pred_test)

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f} | Acc: {acc:.2f}% | Test Loss: {loss_test:.5f} | Test Acc: {acc_test:.2f}%")

    # Save Model
    save = True
    if save:
        torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
        print(f"Model {MODEL_NAME} has been saved in {MODEL_SAVE_PATH}")

    # Visualize Loss curve
    # plt.title("Train Loss")
    # plt.plot(torch.arange(0, len(loss_metrics), 1), loss_metrics, c='b')
    # plt.show()

    # Visualize Decision Boundary
    # visualize_decision_boundary_3d(model, X_train[:40], y_train[:40], predictions=y_pred_real)

# Test Model
# X_real = torch.tensor([
#     [0, 0, 128],  # Dark
#     [197, 225, 165],  # Light
#     [186, 104, 200],  # Light
#     [74, 20, 140],  # Dark
#     [173, 20, 87],  # Dark
#     [96, 165, 250],  # Light
#     [126, 34, 206],  # Dark
# ], dtype=torch.float32)
# # Normalize X_real
# X_real = X_real / 255
# with torch.inference_mode():
#     model.eval()
#     y_logits_real = model(X_real)
#     y_pred_real = torch.round(torch.sigmoid(y_logits_real))
#
#     print(y_pred_real)
