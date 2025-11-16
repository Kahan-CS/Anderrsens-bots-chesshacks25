import torch
import torch.nn as nn
import torch.nn.functional as F
import chess



# ============================================================================
# BOARD ENCODING (must match main.py exactly - 18 channels)
# ============================================================================
class BoardEncoder:
    """18-channel board encoding matching main.py"""

    @staticmethod
    def encode_board(board: chess.Board) -> torch.Tensor:
        """Convert board to tensor (18, 8, 8)"""
        planes = torch.zeros(18, 8, 8, dtype=torch.float32)

        piece_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

        # Encode piece positions (channels 0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)

                plane_idx = piece_idx[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane_idx += 6

                planes[plane_idx, rank, file] = 1.0

        # Channel 12: Repetition (placeholder)
        planes[12, :, :] = 0.0

        # Channel 13: Turn
        planes[13, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

        # Channels 14-17: Castling rights
        planes[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        planes[16, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        planes[17, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

        return planes


class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ValueModel(nn.Module):
    def __init__(self, channels=128, num_blocks=12):
        super().__init__()

        self.conv_in = nn.Conv2d(18, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.conv_value = nn.Conv2d(channels, 8, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(8)

        self.fc_value1 = nn.Linear(8 * 8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.blocks(x)

        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = self.fc_value2(v)

        return v
