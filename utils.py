"""
Some utils methods
"""

from pathlib import Path
import torch


def accuracy_fn(y_true, y_pred):
    """
    Get Accuracy

    :param y_true:
    :param y_pred:
    :return:
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc


def get_model_name() -> str:
    """
    Get the model Name

    :return:
    """
    return "model_colors.pt"


def get_model_path() -> str:
    """
    Get Model path

    :return:
    """
    path = Path("models")
    path.mkdir(parents=True, exist_ok=True)
    name = get_model_name()
    return path / name


def get_truth_table() -> dict:
    """
    Get the Truth Table

    :return:
    """
    return {
        0: 'Bright',
        1: 'Dark'
    }
