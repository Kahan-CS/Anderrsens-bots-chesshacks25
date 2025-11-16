# engine/search.py
import chess
from src.move_index import move_to_index

class SimpleSearch:
    def __init__(self, policy_fn, value_fn=None, top_moves_fn=None,
                 depth=2):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.top_moves_fn = top_moves_fn
        self.depth = depth
        # Add this line:
        self.move_index_fn = move_to_index

    def pick_move(self, board: chess.Board):
        legal = list(board.legal_moves)
        if not legal:
            return None

        policy = self.policy_fn(board)

        scored = []
        for mv in legal:
            idx = move_to_index(mv, board)
            scored.append((mv, policy[idx]))

        # sort by policy prior
        scored.sort(key=lambda x: x[1], reverse=True)

        # print top moves
        if self.top_moves_fn:
            print("\nEngine top-policy moves:")
            for mv, p in self.top_moves_fn(board, 10):
                print(f"  {mv.uci()}   {p:.4f}")

        # value-guided search
        best_move = None
        best_score = -9999

        for mv, prior in scored[:12]:   # top 12 moves
            board.push(mv)
            score = -self.search(board, self.depth - 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = mv

        return best_move

    def search(self, board, depth):
        if depth == 0 or board.is_game_over():
            if self.value_fn is None:
                return 0.0
            return self.value_fn(board)

        best = -9999
        for mv in board.legal_moves:
            board.push(mv)
            score = -self.search(board, depth - 1)
            board.pop()
            best = max(best, score)

        return best
