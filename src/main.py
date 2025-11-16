import torch
import chess

from src.engine import CombinedEngine
from src.load_policy import load_policy_model
from src.load_value import load_value_model

from src.utils import chess_manager, GameContext
from chess import Move

# Import the correct move index function
from src.move_index import move_to_index


print("Loading models...")

device = "cuda" if torch.cuda.is_available() else "cpu"

policy_model = load_policy_model(device)
value_model  = load_value_model(device)

engine = CombinedEngine(policy_model, value_model, device=device)

print("✓ Chess bot initialized")


@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    board = ctx.board
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available.")

    try:
        # Engine chooses move
        move = engine.choose_move(board)

        # Get policy probabilities
        probs = engine.policy.get_policy(board)
        legal_dict = {}

        for mv in legal_moves:
            idx = move_to_index(mv, board)    # <-- FIXED
            legal_dict[str(mv)] = float(probs[idx])

        # Log to UI
        ctx.logProbabilities(legal_dict)

        print(move)
        return move

    except Exception as e:
        print("[ERROR] Engine failure:", e)
        ctx.logProbabilities({})
        return legal_moves[0]


@chess_manager.reset
def reset_game(ctx: GameContext):
    pass
