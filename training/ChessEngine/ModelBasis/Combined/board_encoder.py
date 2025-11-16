# engine/board_encoder.py
import numpy as np
import chess

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board: chess.Board):
    """
    EXACT encoder used during dataset generation.
    Produces a (17,8,8) tensor.
    """

    planes = np.zeros((17, 8, 8), dtype=np.float32)

    # == PIECE PLANES ==
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        base = 0 if piece.color == chess.WHITE else 6
        plane = base + PIECE_TO_PLANE[piece.piece_type]
        planes[plane, row, col] = 1.0

    # == SIDE TO MOVE (plane 12): 1 if BLACK to move ==
    if board.turn == chess.BLACK:
        planes[12,:,:] = 1.0

    # == CASTLING PLANES ==
    planes[13,:,:] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14,:,:] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15,:,:] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16,:,:] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    return planes
