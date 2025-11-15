# engine/move_index.py
import chess

# 8 sliding directions (R,B,Q)
DIRECTIONS = [
    (1, 0),  (-1, 0),  (0, 1),  (0, -1),
    (1, 1),  (1, -1), (-1, 1), (-1, -1)
]

# Promotion piece order MUST match training
PROMO_ORDER = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """
    EXACT AlphaZero-style 4672 indexing as used in your training data.
    73 moves per from-square.

    Layout per from-square:
      0–55  : sliding moves (8 dirs × 7 squares)
      56–72 : promotion lanes (17 total)
      (all else fallback = 0 bucket)
    """

    from_sq = move.from_square
    to_sq = move.to_square

    fx = chess.square_file(from_sq)
    fy = chess.square_rank(from_sq)
    tx = chess.square_file(to_sq)
    ty = chess.square_rank(to_sq)

    dx = tx - fx
    dy = ty - fy

    # ============================================================
    # 1. PROMOTIONS  (MUST MATCH TRAINING EXACTLY)
    # ============================================================
    if move.promotion is not None:
        promo_type = PROMO_ORDER.index(move.promotion)  # 0..3

        # Training used: 56 + (rank_diff * 7) + promo_type
        # rank_diff = (ty - fy)
        direction_idx = 56 + (ty - fy) * 7 + promo_type

        return from_sq * 73 + direction_idx

    # ============================================================
    # 2. SLIDING MOVES (8 directions × 7 distances = 56)
    # ============================================================
    move_dir = None
    for i, (dx_dir, dy_dir) in enumerate(DIRECTIONS):

        # Must align with the direction vector
        if dx_dir != 0 and dx % dx_dir != 0:
            continue
        if dy_dir != 0 and dy % dy_dir != 0:
            continue

        # Calculate number of squares moved
        if dx_dir != 0:
            k = dx // dx_dir
        else:
            k = dy // dy_dir

        if k > 0:
            move_dir = i * 7 + (k - 1)
            break

    if move_dir is None:
        # Training code: "Knight or illegal → encode as zero bucket"
        return from_sq * 73   # bucket 0 for this from-square

    return from_sq * 73 + move_dir


# ----------------------------------------------------------------
# Optional reverse mapping helper (index → Move)
# ----------------------------------------------------------------
def index_to_move(board: chess.Board, index: int) -> chess.Move:
    """
    Reverse mapping for debugging or visualization.
    Works exactly with the training scheme.
    """
    from_sq = index // 73
    offset = index % 73

    # Promotion?
    if offset >= 56:
        promo_block = offset - 56
        rank_diff = promo_block // 7
        promo_type = PROMO_ORDER[promo_block % 4]

        fx = chess.square_file(from_sq)
        fy = chess.square_rank(from_sq)
        ty = fy + rank_diff
        tx_candidates = [i for i in range(8)]

        # Find valid to-square
        for tx in tx_candidates:
            to_sq = chess.square(tx, ty)
            if board.is_pseudo_legal(chess.Move(from_sq, to_sq, promo_type)):
                return chess.Move(from_sq, to_sq, promo_type)

        # fallback
        return chess.Move.null()

    # Sliding move
    dir_idx = offset // 7
    dist = (offset % 7) + 1

    dx, dy = DIRECTIONS[dir_idx]
    fx = chess.square_file(from_sq)
    fy = chess.square_rank(from_sq)
    tx = fx + dx * dist
    ty = fy + dy * dist
    if 0 <= tx < 8 and 0 <= ty < 8:
        return chess.Move(from_sq, chess.square(tx, ty))

    # fallback
    return chess.Move.null()
