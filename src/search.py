# engine/search.py

import time
import math
from typing import Optional, Dict, Tuple, List

import chess
from src.move_index import move_to_index


class MCTSNode:
    """
    Single node in the MCTS tree.
    Q, N are stored from the ROOT player's perspective.
    """
    def __init__(self, prior: float = 1.0):
        # Prior of this node from its parent edge (not heavily used for root).
        self.P = prior

        # Node statistics (from root's perspective)
        self.N = 0          # total visits to this node
        self.W = 0.0        # total value of simulations
        self.Q = 0.0        # mean value (= W / N)

        # Expansion state
        self.is_expanded = False

        # Per-move stats (edge statistics)
        self.priors: Dict[chess.Move, float] = {}   # P(s,a)
        self.Nsa: Dict[chess.Move, int] = {}        # N(s,a)
        self.Wsa: Dict[chess.Move, float] = {}      # W(s,a)
        self.Qsa: Dict[chess.Move, float] = {}      # Q(s,a)
        self.children: Dict[chess.Move, "MCTSNode"] = {}  # s,a -> child node


class SimpleSearch:
    """
    Monte Carlo Tree Search engine:

    - Policy-guided MCTS (PUCT)
    - Optional value head for leaf evaluation
    - Time-based playout budget
    - Same public interface as previous alpha-beta version
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
        cpuct: float = 1.5,
    ):
        """
        policy_fn(board) -> policy vector over all moves (flat array).
        value_fn(board)  -> scalar value in [-1, 1] from the SIDE-TO-MOVE perspective.
        top_moves_fn(board, k) -> [(move, prob), ...] for debug printing.

        depth:
            Used as a soft cap on simulation depth in plies (0 = root).
        policy_root_weight:
            Unused in pure MCTS logic, kept for interface compatibility.
        use_value:
            If False, the value head is ignored and a simple neutral evaluation (0)
            is used at non-terminal leaves (search is then more rollout-like).
        cpuct:
            Exploration constant for PUCT; higher = more exploration.
        """
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.top_moves_fn = top_moves_fn
        self.max_depth = depth

        self.policy_root_weight = policy_root_weight
        self.use_value = use_value
        self.cpuct = cpuct

        # Move index mapping for policy
        self.move_index_fn = move_to_index

        # Tree & search state
        self.root: Optional[MCTSNode] = None
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
        Choose a move using MCTS within `time_limit` seconds.
        `max_depth` (if given) overrides the constructor's `depth` as
        the maximum plies per simulation.
        """
        if max_depth is not None:
            self.max_depth = max_depth

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Reset search context
        self.nodes = 0
        self.time_limit = time_limit
        self.start_time = time.time()
        self.stop = False

        # Optional: print top-policy moves (pure policy, no search)
        if self.top_moves_fn:
            priors_for_print = self._compute_policy_priors(board, legal_moves)
            print("\nEngine top-policy moves:")
            for mv, p in self.top_moves_fn(board, 10):
                # in case top_moves_fn is independent from priors_for_print
                print(f"  {mv.uci()}   {p:.4f}")

        # Initialize root node; we track values from root player's perspective
        self.root = MCTSNode(prior=1.0)

        # Run simulations until we run out of time
        while not self._out_of_time():
            # We must operate on a copy of the board for each simulation
            b_copy = board.copy()
            self._run_simulation(b_copy, self.root)

        # After search, choose move with highest visit count at root
        root = self.root
        legal = list(board.legal_moves)

        if root is None or not root.is_expanded:
            # Fallback to pure policy if something went wrong
            priors = self._compute_policy_priors(board, legal)
            return max(legal, key=lambda m: priors[m])

        best_move = None
        best_visits = -1

        for mv in legal:
            visits = root.Nsa.get(mv, 0)
            if visits > best_visits:
                best_visits = visits
                best_move = mv

        # Fallback if no visits (shouldn't really happen)
        if best_move is None:
            priors = self._compute_policy_priors(board, legal)
            best_move = max(legal, key=lambda m: priors[m])

        return best_move

    # ======================================================================
    # MCTS CORE
    # ======================================================================
    def _run_simulation(self, board: chess.Board, root_node: MCTSNode):
        """
        One full MCTS simulation from root:
            selection -> expansion -> evaluation -> backup
        All Q, W, N are stored in root-player perspective.
        """
        node = root_node
        path: List[Tuple[MCTSNode, Optional[chess.Move]]] = []
        depth = 0

        # --- SELECTION & EXPANSION --------------------------------------
        while True:
            if self._out_of_time():
                return

            # If game is over, evaluate terminal and stop
            if board.is_game_over():
                leaf_value_side = self._terminal_value_from_side_to_move(board)
                break

            # Depth limit: treat as leaf (no further expansion)
            if self.max_depth is not None and depth >= self.max_depth:
                leaf_value_side = self._leaf_value_from_side_to_move(board)
                break

            if not node.is_expanded:
                # Expand this node and evaluate with NN (or heuristic)
                leaf_value_side = self._expand_node(node, board)
                break

            # Otherwise, select a child via PUCT
            move = self._select_child_move(node)
            path.append((node, move))

            board.push(move)
            depth += 1

            child = node.children.get(move)
            if child is None:
                # Create a new (unexpanded) child node with the edge prior
                prior = node.priors.get(move, 0.0)
                child = MCTSNode(prior=prior)
                node.children[move] = child

            node = child

        # leaf_value_side: value from perspective of SIDE TO MOVE at leaf position
        # Convert to root player's perspective:
        # if an odd number of plies from root, sign flip
        root_value = leaf_value_side * ((-1) ** depth)

        # --- BACKUP ------------------------------------------------------
        # Update stats along the path (from root to leaf)
        # Path contains all (node, move) pairs along edges.
        for n, mv in path:
            n.N += 1
            n.W += root_value
            n.Q = n.W / n.N

            if mv is not None:
                n.Nsa[mv] = n.Nsa.get(mv, 0) + 1
                n.Wsa[mv] = n.Wsa.get(mv, 0.0) + root_value
                n.Qsa[mv] = n.Wsa[mv] / n.Nsa[mv]

        # Also count visit for the final node (leaf)
        node.N += 1
        node.W += root_value
        node.Q = node.W / node.N

        self.nodes += 1

    def _expand_node(self, node: MCTSNode, board: chess.Board) -> float:
        """
        Expand node:
            - get legal moves
            - compute policy priors for these moves
            - (optionally) evaluate value from value head
        Returns value from SIDE-TO-MOVE perspective at this node.
        """
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            # No legal moves: terminal; delegate to terminal evaluator.
            node.is_expanded = True
            return self._terminal_value_from_side_to_move(board)

        # Compute policy priors for all legal moves, normalized
        priors = self._compute_policy_priors(board, legal_moves)

        node.priors = priors
        node.is_expanded = True

        # Value from side-to-move perspective
        leaf_value = self._leaf_value_from_side_to_move(board)
        return leaf_value

    def _select_child_move(self, node: MCTSNode) -> chess.Move:
        """
        PUCT selection:
            a* = argmax_a [ Q(s,a) + cpuct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a)) ]
        All Q values are from root player's perspective.
        """
        cpuct = self.cpuct

        # Total edge visits from this node
        total_N = sum(node.Nsa.get(m, 0) for m in node.priors.keys())
        sqrt_total_N = math.sqrt(total_N + 1e-8)

        best_score = -float("inf")
        best_move = None

        for mv, prior in node.priors.items():
            Nsa = node.Nsa.get(mv, 0)
            Qsa = node.Qsa.get(mv, 0.0)

            U = cpuct * prior * sqrt_total_N / (1 + Nsa)
            score = Qsa + U

            if score > best_score:
                best_score = score
                best_move = mv

        # In degenerate cases, just pick some legal move
        if best_move is None:
            best_move = next(iter(node.priors.keys()))

        return best_move

    # ======================================================================
    # EVALUATION HELPERS
    # ======================================================================
    def _compute_policy_priors(self, board: chess.Board, legal_moves):
        """
        Map full policy vector to legal moves and renormalize.
        Returns a dict: move -> prior (sum = 1).
        """
        priors = {}

        if self.policy_fn is None:
            # Uniform prior over legal moves
            n = len(legal_moves)
            if n == 0:
                return {}
            p = 1.0 / n
            for mv in legal_moves:
                priors[mv] = p
            return priors

        policy = self.policy_fn(board)  # assume flat vector over all moves

        total = 0.0
        for mv in legal_moves:
            idx = self.move_index_fn(mv, board)
            p = float(policy[idx])
            # clip negatives just in case
            if p < 0.0:
                p = 0.0
            priors[mv] = p
            total += p

        if total <= 0.0:
            # If the NN gave all zeros, fall back to uniform
            n = len(legal_moves)
            p = 1.0 / n
            for mv in legal_moves:
                priors[mv] = p
            return priors

        # Normalize
        for mv in legal_moves:
            priors[mv] /= total

        return priors

    def _terminal_value_from_side_to_move(self, board: chess.Board) -> float:
        """
        Returns value from perspective of SIDE TO MOVE at this terminal state:
            checkmated side -> -1
            draw           -> 0
        """
        if not board.is_game_over():
            return 0.0

        if board.is_checkmate():
            # Side to move is checkmated -> loss
            return -1.0

        # Stalemate, repetition, 50-move rule, etc.
        return 0.0

    def _leaf_value_from_side_to_move(self, board: chess.Board) -> float:
        """
        Non-terminal leaf evaluation from SIDE-TO-MOVE perspective.
        Uses value head if available and enabled; otherwise returns 0.
        """
        # If the game is already over, defer to terminal evaluator
        if board.is_game_over():
            return self._terminal_value_from_side_to_move(board)

        if not self.use_value or self.value_fn is None:
            return 0.0

        v = self.value_fn(board)

        # Support torch / numpy / float
        try:
            import torch  # type: ignore

            if isinstance(v, torch.Tensor):
                v = float(v.detach().cpu().item())
        except Exception:
            pass

        try:
            v = float(v)
        except Exception:
            v = 0.0

        # Optionally clamp to [-1, 1]
        if v > 1.0:
            v = 1.0
        elif v < -1.0:
            v = -1.0

        return v

    # ======================================================================
    # TIME CONTROL
    # ======================================================================
    def _out_of_time(self) -> bool:
        if self.time_limit is None:
            return False
        if self.stop:
            return True
        if (time.time() - self.start_time) >= self.time_limit:
            self.stop = True
        return self.stop
