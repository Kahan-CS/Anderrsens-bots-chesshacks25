# engine/policy_wrapper.py
import torch
import chess
import numpy as np
from Combined.board_encoder import board_to_tensor
from Combined.move_index import move_to_index

class PolicyWrapper:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def get_policy(self, board: chess.Board):
        x = board_to_tensor(board)
        x = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # mask illegals
        mask = np.zeros_like(probs, dtype=np.float32)
        for mv in board.legal_moves:
            idx = move_to_index(mv, board)
            mask[idx] = 1.0

        probs = probs * mask
        s = probs.sum()
        if s <= 1e-12:
            legal = np.where(mask == 1.0)[0]
            probs[legal] = 1.0 / len(legal)
        else:
            probs /= s

        return probs

    def get_top_moves(self, board: chess.Board, N=5):
        """
        Returns list of (move, probability) sorted by prob desc.
        """
        probs = self.get_policy(board)
        moves = list(board.legal_moves)

        scored = []
        for mv in moves:
            idx = move_to_index(mv, board)
            scored.append((mv, probs[idx]))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:N]
