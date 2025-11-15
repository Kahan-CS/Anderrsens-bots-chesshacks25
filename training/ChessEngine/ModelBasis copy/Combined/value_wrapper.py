# engine/value_wrapper.py
import torch
import numpy as np
from Value.value_model import ValueModel, BoardEncoder

class ValueWrapper:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, board):
        # value model uses its OWN encoder (18 channels!)
        x = BoardEncoder.encode_board(board)
        x = x.unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.model(x).cpu().numpy()[0][0]

        return float(value)
