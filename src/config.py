import os
from pathlib import Path

# Camera device configuration
CAMERA_DEVICE = 0

def resource_path(relative_path):
    """
    Get the absolute path to a resource file.
    Works for both development and PyInstaller builds.
    """
    base_path = os.path.abspath("GUI")
    return os.path.join(base_path, relative_path)

# UI file path
MAIN_GUI = resource_path("Main.ui")

# Model paths
MODEL_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
FACIAL_EXPRESSION_MODEL = os.path.join(MODEL_WEIGHTS_DIR, "facial_expression_model_weights.h5")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

# Ensure directories exist
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
