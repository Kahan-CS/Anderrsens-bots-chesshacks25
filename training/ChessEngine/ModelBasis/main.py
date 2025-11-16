# main.py
import chess
from Combined.engine import CombinedEngine

# import your models
from Policy.policy_model import PolicyNetRes      # your policy model
# from value_model import ValueNet            # optional

import torch


def load_policy_model(device="cpu"):
    model = PolicyNetRes()
    model.load_state_dict(torch.load("/Users/devonrempel/PycharmProjects/ModelBasis/Policy/policy_resnet.pt", map_location=device))
    model.eval()
    return model.to(device)


def load_value_model(device="cpu"):
    """
    Loads the trained value model from disk.
    Returns: ValueModel() on the correct device, or None if loading fails.
    """

    import torch
    import os
    from Value.value_model import ValueModel

    VALUE_PATH = "/Users/devonrempel/PycharmProjects/ModelBasis/Value/value_model.pth"

    if not os.path.exists(VALUE_PATH):
        print(f"[ValueModel] No file found at {VALUE_PATH}. Value model disabled.")
        return None

    try:
        model = ValueModel()
        state = torch.load(VALUE_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print("[ValueModel] Loaded successfully.")
        return model.to(device)

    except Exception as e:
        print("[ValueModel] Failed to load:", e)
        return None



def print_board(board):
    print("\n" + str(board) + "\n")
    print("FEN:", board.fen(), "\n")


def get_user_move(board):
    """
    Get user move in SAN or UCI.
    Returns a chess.Move or None if input invalid.
    """

    while True:
        user = input("Your move (or 'quit'): ").strip()

        if user.lower() in ["quit", "exit"]:
            return None

        # Try SAN first ("Nf3", "exd5", "O-O", etc)
        try:
            move = board.parse_san(user)
            return move
        except Exception:
            pass

        # Try UCI ("e2e4", "g8f6", etc)
        try:
            move = chess.Move.from_uci(user)
            if move in board.legal_moves:
                return move
        except Exception:
            pass

        print("Invalid move. Try again.")


def choose_side():
    while True:
        side = input("Play as (w)hite or (b)lack? ").strip().lower()
        if side in ["w", "white"]:
            return chess.WHITE
        if side in ["b", "black"]:
            return chess.BLACK
        print("Invalid choice. Please enter 'w' or 'b'.")


# ============================================================
#                   PLAY A GAME
# ============================================================

if __name__ == "__main__":
    device = "cpu"

    print("Loading models...")
    policy_model = load_policy_model(device)
    value_model = load_value_model(device)  # currently None

    engine = CombinedEngine(policy_model, value_model, device=device)

    board = chess.Board()
    print_board(board)

    human_color = choose_side()
    print(f"You are playing {'White' if human_color else 'Black'}.")

    # main game loop
    while not board.is_game_over():

        if board.turn == human_color:
            # Human move
            move = get_user_move(board)
            if move is None:
                print("Exiting game.")
                break

            board.push(move)
            print_board(board)

        else:
            # Engine move
            print("Engine thinking...")
            move = engine.choose_move(board)

            if move is None:
                print("Engine resigns or no legal moves.")
                break

            print(f"Engine plays: {board.san(move)}")
            board.push(move)
            print_board(board)

    print("Game over!")
    print("Result:", board.result())
