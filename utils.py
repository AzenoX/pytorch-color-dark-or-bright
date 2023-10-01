import torch
import numpy as np
import matplotlib.pyplot as plt


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc


def visualize_decision_boundary_3d(model, X, y, predictions=None, grid_density=20, subsample_ratio=0.1):
    # Create a meshgrid of RGB color values with reduced density
    r_values = np.linspace(0, 1, grid_density)
    g_values = np.linspace(0, 1, grid_density)
    b_values = np.linspace(0, 1, grid_density)
    r_mesh, g_mesh, b_mesh = np.meshgrid(r_values, g_values, b_values)

    # Flatten the meshgrid arrays
    r_flat = r_mesh.flatten()
    g_flat = g_mesh.flatten()
    b_flat = b_mesh.flatten()

    # Combine the RGB values into a single array
    color_grid = np.column_stack((r_flat, g_flat, b_flat))

    # Randomly subsample points for visualization
    num_samples = int(subsample_ratio * len(color_grid))
    selected_indices = np.random.choice(len(color_grid), num_samples, replace=False)
    subsampled_color_grid = color_grid[selected_indices]

    # Get model predictions for the subsampled color grid
    if predictions is None:
        with torch.no_grad():
            model.eval()
            logits = model(torch.from_numpy(subsampled_color_grid).type(torch.float32))
            predictions = torch.round(torch.sigmoid(logits))

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points based on model predictions
    for i, prediction in enumerate(predictions):
        if prediction == 0:
            ax.scatter(subsampled_color_grid[i, 0], subsampled_color_grid[i, 1], subsampled_color_grid[i, 2], c='green', marker='.')
        else:
            ax.scatter(subsampled_color_grid[i, 0], subsampled_color_grid[i, 1], subsampled_color_grid[i, 2], c='green', marker='.')

    # Plot training points
    for i, label in enumerate(y):
        if label == 0:
            ax.scatter(X[i, 0], X[i, 1], X[i, 2], c='blue', marker='o', s=60, edgecolor='k', label='Bright')
        else:
            ax.scatter(X[i, 0], X[i, 1], X[i, 2], c='red', marker='o', s=60, edgecolor='k', label='Dark')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('Decision Boundary in RGB Color Space')
    ax.legend()

    plt.show()
