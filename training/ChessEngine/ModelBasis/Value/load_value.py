# Combined/load_value.py

import torch
import os
from Value.value_model import ValueModel

VALUE_PATH = "/Users/devonrempel/PycharmProjects/ModelBasis/Value/value_model.pth"

def load_value_model(device="cpu"):
    """
    Loads the trained value model if it exists.
    Returns None if not found (engine will run policy-only).
    """

    if not os.path.exists(VALUE_PATH):
        print(f"[ValueModel] No value model found at {VALUE_PATH}. Using policy-only engine.")
        return None

    try:
        model = ValueModel()
        state = torch.load(VALUE_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print("[ValueModel] Loaded successfully.")
        return model.to(device)

    except Exception as e:
        print("[ValueModel] Error loading model:", e)
        return None
