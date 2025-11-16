# engine/search.py

import time
import math
from typing import Optional

import chess
from src.move_index import move_to_index


class SimpleSearch:
    """
    Strong neural-guided search engine:
    - Iterative Deepening
    - Negamax + Alpha-Beta
    - Transposition Table (keyed by FEN)
    - Policy-based move ordering at root
    - Policy-biased root scoring (policy can override noisy value)
    - MVV-LVA capture ordering inside the tree
    """

    MATE_SCORE = 100000

    def __init__(
        self,
        policy_fn,
        value_fn=None,
        top_moves_fn=None,
        depth: int = 4,
        policy_root_weight: float = 0.8,
        use_value: bool = True,
    ):
        """
        policy_root_weight:
            How much policy prior influences the FINAL root decision.
            Higher -> closer to pure policy, lower -> closer to pure value search.
        use_value:
            If False, disable the value head completely (engine becomes policy-guided
            with a search skeleton, but evaluation is always 0 except mates).
        """
        # depth = maximum search depth for iterative deepening
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.top_moves_fn = top_moves_fn
        self.max_depth = depth

        self.policy_root_weight = policy_root_weight
        self.use_value = use_value

        # Move index mapping for policy
        self.move_index_fn = move_to_index

        # Transposition table: key (FEN) -> {depth, value, flag, best_move}
        self.tt = {}

        # Search state
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit: Optional[float] = None
        self.stop = False

    # ======================================================================
    # PUBLIC API
    # ======================================================================
    def pick_move(
        self,
        board: chess.Board,
        time_limit: float = 5.0,
        max_depth: Optional[int] = None,
    ) -> Optional[chess.Move]:
        """
        Choose a move using iterative deepening up to max_depth (default: self.max_depth)
        and within time_limit seconds.
        """
        if max_depth is None:
            max_depth = self.max_depth

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Reset search context
        self.tt.clear()
        self.nodes = 0
        self.time_limit = time_limit
        self.start_time = time.time()
        self.stop = False

        # Root policy priors for move ordering and bias
        priors = self._compute_policy_priors(board, legal_moves)

        # Optional: print top-policy moves
        if self.top_moves_fn:
            print("\nEngine top-policy moves:")
            for mv, p in self.top_moves_fn(board, 10):
                print(f"  {mv.uci()}   {p:.4f}")

        best_move: Optional[chess.Move] = None
        best_score = -math.inf
        previous_pv_move: Optional[chess.Move] = None

        # ============================================================
        # ITERATIVE DEEPENING
        # ============================================================
        for depth in range(1, max_depth + 1):
            if self.stop:
                break

            score, move = self._search_root(
                board,
                depth,
                priors,
                previous_pv_move,
            )

            if self.stop:
                break

            if move is not None:
                best_move = move
                best_score = score
                previous_pv_move = move

            # Debug if you want:
            # print(f"[Search] depth={depth} score={score:.3f} nodes={self.nodes}")

        # Fallback if we never completed depth 1
        if best_move is None:
            best_move = max(legal_moves, key=lambda m: priors[m])

        return best_move

    # ======================================================================
    # ROOT LEVEL SEARCH
    # ======================================================================
    def _search_root(
        self,
        board: chess.Board,
        depth: int,
        priors: dict,
        previous_pv_move: Optional[chess.Move],
    ):
        alpha = -math.inf
        beta = math.inf

        legal_moves = list(board.legal_moves)

        # Order: previous PV move first (if still legal), then by policy prior
        ordered: list[chess.Move] = []
        if previous_pv_move in legal_moves:
            ordered.append(previous_pv_move)
            legal_moves.remove(previous_pv_move)

        remaining = sorted(legal_moves, key=lambda m: priors[m], reverse=True)
        ordered.extend(remaining)

        best_score = -math.inf
        best_move: Optional[chess.Move] = None

        for mv in ordered:
            if self._out_of_time():
                break

            board.push(mv)
            search_score = -self._search(board, depth - 1, -beta, -alpha)
            board.pop()

            if self.stop:
                break

            # ---- ROOT-LEVEL POLICY BIAS --------------------------------
            # combined_score = value-search-score + policy bonus
            prior = priors.get(mv, 0.0)
            combined_score = search_score + self.policy_root_weight * prior

            if combined_score > best_score:
                best_score = combined_score
                best_move = mv

            # Alpha-beta window at root is based on *search* score,
            # so we don't feed the policy bias back into the tree.
            if search_score > alpha:
                alpha = search_score

        return best_score, best_move

    # ======================================================================
    # CORE SEARCH (NEGAMAX + ALPHA-BETA + TRANSPOSITION TABLE)
    # ======================================================================
    def _search(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        if self._out_of_time():
            return self._evaluate(board)

        self.nodes += 1

        # Leaf / terminal
        if depth <= 0 or board.is_game_over():
            return self._evaluate(board)

        # --- TRANSPOSITION TABLE LOOKUP --------------------------------
        key = board.fen()  # robust in any version: use FEN as TT key
        tt_entry = self.tt.get(key)

        if tt_entry is not None and tt_entry["depth"] >= depth:
            flag = tt_entry["flag"]
            val = tt_entry["value"]

            if flag == "EXACT":
                return val
            elif flag == "LOWER":
                alpha = max(alpha, val)
            elif flag == "UPPER":
                beta = min(beta, val)

            if alpha >= beta:
                return val

        alpha_orig = alpha
        best_value = -math.inf
        best_move: Optional[chess.Move] = None

        # MOVE ORDERING
        legal = list(board.legal_moves)
        ordered: list[chess.Move] = []

        # 1) TT best move first (if available and legal)
        if tt_entry and tt_entry["best_move"] in legal:
            tt_best = tt_entry["best_move"]
            ordered.append(tt_best)
            legal.remove(tt_best)

        # 2) Captures (MVV-LVA) then quiet moves
        captures = []
        quiets = []
        for mv in legal:
            if board.is_capture(mv):
                captures.append(mv)
            else:
                quiets.append(mv)

        def mvv_lva(mv: chess.Move) -> int:
            victim = board.piece_at(mv.to_square)
            attacker = board.piece_at(mv.from_square)
            if victim is None or attacker is None:
                return 0
            val = {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
                chess.KING: 20000,
            }
            return val[victim.piece_type] * 10 - val[attacker.piece_type]

        captures.sort(key=mvv_lva, reverse=True)
        ordered.extend(captures)
        ordered.extend(quiets)

        # NEGAMAX LOOP
        for mv in ordered:
            board.push(mv)
            score = -self._search(board, depth - 1, -beta, -alpha)
            board.pop()

            if self.stop:
                # Just propagate something upwards so recursion unwinds
                return alpha

            if score > best_value:
                best_value = score
                best_move = mv

            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break  # beta cutoff

        # --- WRITE TO TRANSPOSITION TABLE ----------------------------
        if best_value <= alpha_orig:
            flag = "UPPER"
        elif best_value >= beta:
            flag = "LOWER"
        else:
            flag = "EXACT"

        self.tt[key] = {
            "depth": depth,
            "value": best_value,
            "flag": flag,
            "best_move": best_move,
        }

        return best_value

    # ======================================================================
    # INTERNAL HELPERS
    # ======================================================================
    def _compute_policy_priors(self, board: chess.Board, legal_moves):
        """Policy is optional but powerful for move ordering at the root."""
        priors = {}
        if self.policy_fn is None:
            for mv in legal_moves:
                priors[mv] = 0.0
            return priors

        policy = self.policy_fn(board)  # assume flat vector over all moves
        for mv in legal_moves:
            idx = self.move_index_fn(mv, board)
            priors[mv] = float(policy[idx])

        return priors

    def _evaluate(self, board: chess.Board) -> float:
        """Exact terminal scores + (optionally) value net."""
        if board.is_game_over():
            if board.is_checkmate():
                # Side to move is checkmated => big negative score
                return -self.MATE_SCORE
            # Stalemate, repetition, etc.
            return 0.0

        if not self.use_value or self.value_fn is None:
            # Ignore value head entirely if disabled
            return 0.0

        v = self.value_fn(board)

        # Try to be robust to torch / numpy / float
        try:
            import torch

            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
        except Exception:
            pass

        try:
            return float(v)
        except Exception:
            return 0.0

    def _out_of_time(self) -> bool:
        if self.time_limit is None:
            return False
        if self.stop:
            return True
        if (time.time() - self.start_time) >= self.time_limit:
            self.stop = True
        return self.stop
