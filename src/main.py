# main.py — Chess Bot Entry Point for ChessHacks Platform

import torch
import chess

# Import your engine components
from training.ChessEngine.ModelBasis.Combined.engine import CombinedEngine
from training.ChessEngine.ModelBasis.Policy.load_policy import load_policy_model
from training.ChessEngine.ModelBasis.Value.load_value import load_value_model   # your loader

# ChessHacks platform interface
from .utils import chess_manager, GameContext
from chess import Move


# ============================================================================
# INITIALIZATION (runs once)
# ============================================================================

print("Loading models...")

device = "cuda" if torch.cuda.is_available() else "cpu"

policy_model = load_policy_model(device)
value_model  = load_value_model(device)   # may return None → policy-only

engine = CombinedEngine(policy_model, value_model, device=device)

print("✓ Chess bot initialized")


# ============================================================================
# ENTRYPOINT (called every time the bot must move)
# ============================================================================

@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    """
    Entrypoint required by ChessHacks platform.

    Args:
        ctx: GameContext containing the current board.

    Returns:
        python-chess Move
    """

    board = ctx.board
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available.")

    try:
        # -----------------------------
        # Engine chooses move
        # -----------------------------
        move = engine.choose_move(board)

        # -----------------------------
        # Log probabilities (for UI)
        # -----------------------------
        probs = engine.policy.get_policy(board)
        legal_dict = {}

        for mv in legal_moves:
            idx = engine.searcher.move_index_fn(mv, board)
            legal_dict[str(mv)] = float(probs[idx])

        # log to ChessHacks
        ctx.logProbabilities(legal_dict)

        return move

    except Exception as e:
        print("[ERROR] Engine failure:", e)
        ctx.logProbabilities({})
        return legal_moves[0]  # fallback


# ============================================================================
# RESET (called once at new game)
# ============================================================================

@chess_manager.reset
def reset_game(ctx: GameContext):
    """
    Called at the start of each new game.
    Reset caches or internal search state here.
    """
    # No persistent state to reset (unless you add TT later)
    pass
