# engine/engine.py
import chess
from src.policy_wrapper import PolicyWrapper
from src.value_wrapper import ValueWrapper
from src.search import SimpleSearch
class CombinedEngine:
    def __init__(self, policy_model, value_model=None, device="cpu"):

        self.policy = PolicyWrapper(policy_model, device)

        if value_model is not None:
            self.value = ValueWrapper(value_model, device)
            use_value = True
        else:
            self.value = None
            use_value = False

        self.searcher = SimpleSearch(
            policy_fn=self.policy.get_policy,
            value_fn=self.value.evaluate if use_value else None,
            top_moves_fn=self.policy.get_top_moves,
            depth=3 if use_value else 1
        )

    def choose_move(self, board):
        return self.searcher.pick_move(board)
