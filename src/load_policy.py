# Combined/load_policy.py

import torch
from src.policy_model import PolicyNetRes
import os

from huggingface_hub import hf_hub_download

REPO_ID = "Kahanesque/chesshacks-anderrsens-bot"
policy_path = hf_hub_download(repo_id=REPO_ID, filename="policy_resnet.pt")
POLICY_PATH = "src/policy_resnet.pt"

def load_policy_model(device="cpu"):
    """
    Loads your trained policy network from disk.
    """

    if not os.path.exists(POLICY_PATH):
        raise FileNotFoundError(f"Policy model not found at {POLICY_PATH}")

    model = PolicyNetRes()
    state = torch.load(POLICY_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("[PolicyModel] Loaded successfully.")
    return model.to(device)
