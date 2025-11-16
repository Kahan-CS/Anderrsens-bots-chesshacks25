# main.py - Chess Bot Entry Point for ChessHacks Platform

"""
Chess Bot with Policy Model + Value Model + Minimax Search

Architecture:
1. Policy Model (17 channels) generates top-N candidate moves
2. Value Model (18 channels) evaluates board positions
3. Minimax search selects best move from candidates

Models are trained separately and loaded from weights/ directory
"""

from huggingface_hub import hf_hub_download

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import os

# ============================================================================
# BOARD ENCODING - POLICY MODEL (17 channels)
# ============================================================================
class PolicyBoardEncoder:
    """
    17-channel encoding for policy model (matches friend's training)
    
    Channels:
    - 0-5: White pieces (P, N, B, R, Q, K)
    - 6-11: Black pieces (P, N, B, R, Q, K)
    - 12: Repetition
    - 13: Turn
    - 14-16: Castling rights (WK, WQ, BK+BQ combined)
    """
    
    @staticmethod
    def encode_board(board: chess.Board) -> torch.Tensor:
        """Returns tensor of shape (17, 8, 8)"""
        planes = torch.zeros(17, 8, 8, dtype=torch.float32)
        
        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                plane_idx = piece_idx[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane_idx += 6
                planes[plane_idx, rank, file] = 1.0
        
        # Metadata
        planes[12, :, :] = 0.0  # Repetition
        planes[13, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        planes[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        # Combine both black castling rights into one channel
        planes[16, :, :] = 1.0 if (board.has_kingside_castling_rights(chess.BLACK) or 
                                     board.has_queenside_castling_rights(chess.BLACK)) else 0.0
        
        return planes


# ============================================================================
# BOARD ENCODING - VALUE MODEL (18 channels)
# ============================================================================
class ValueBoardEncoder:
    """
    18-channel encoding for value model (matches our training)
    
    Channels:
    - 0-5: White pieces (P, N, B, R, Q, K)
    - 6-11: Black pieces (P, N, B, R, Q, K)
    - 12: Repetition
    - 13: Turn
    - 14-17: Castling rights (WK, WQ, BK, BQ)
    """
    
    @staticmethod
    def encode_board(board: chess.Board) -> torch.Tensor:
        """Returns tensor of shape (18, 8, 8)"""
        planes = torch.zeros(18, 8, 8, dtype=torch.float32)
        
        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                plane_idx = piece_idx[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane_idx += 6
                planes[plane_idx, rank, file] = 1.0
        
        # Metadata
        planes[12, :, :] = 0.0  # Repetition
        planes[13, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        planes[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        planes[16, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        planes[17, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        return planes


# ============================================================================
# RESIDUAL BLOCK (shared by both models)
# ============================================================================
class ResidualBlock(nn.Module):
    """Residual block for deep network"""
    
    def __init__(self, channels=128, use_bias=True):
        super().__init__()
        # Value model was trained WITH bias, policy model WITHOUT
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

# ============================================================================
# GROUPNORM RESIDUAL BLOCK (for Value Model only)
# ============================================================================
class ResidualBlockGN(nn.Module):
    """Residual block using GroupNorm (32 groups) for stability on large datasets."""

    def __init__(self, channels=128, use_bias=True, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=use_bias)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=use_bias)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


# ============================================================================
# POLICY MODEL
# ============================================================================
class PolicyModel(nn.Module):
    """
    Policy network that predicts move probabilities
    Architecture matches friend's PolicyNetRes (17 channels, 8 residual blocks)
    """
    
    def __init__(self, policy_size=4672):
        super().__init__()
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(17, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Policy residual blocks were trained WITHOUT bias on convs — pass use_bias=False
        self.res_layers = nn.Sequential(
        *[ResidualBlock(128, use_bias=False) for _ in range(8)]
)
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, policy_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Board tensor of shape (batch, 17, 8, 8)
        
        Returns:
            Move logits of shape (batch, policy_size)
        """
        x = self.conv_in(x)
        x = self.res_layers(x)
        x = self.policy_head(x)
        return x
    
    @torch.no_grad()
    def get_top_n_moves(self, board: chess.Board, n: int = 10) -> List[Tuple[chess.Move, float]]:
        """
        Get top N legal moves from the policy model
        
        Args:
            board: Current chess position
            n: Number of candidate moves to return
        
        Returns:
            List of (move, probability) tuples, sorted by probability
        """
        self.eval()
        
        # Encode board (17 channels)
        board_tensor = PolicyBoardEncoder.encode_board(board).unsqueeze(0)

        # Move encoded board to the same device as the model
        device = next(self.parameters()).device
        board_tensor = board_tensor.to(device)

        # Get move logits
        logits = self.forward(board_tensor).squeeze(0)
        probs = F.softmax(logits, dim=0)
        
        # Get all legal moves with their probabilities
        legal_moves = []
        for move in board.legal_moves:
            # Simple encoding: from_square * 64 + to_square
            move_idx = move.from_square * 64 + move.to_square
            if move_idx < len(probs):
                legal_moves.append((move, probs[move_idx].item()))
        
        # If no moves matched (shouldn't happen), use uniform distribution
        if not legal_moves:
            legal_moves = [(move, 1.0 / len(list(board.legal_moves))) 
                          for move in board.legal_moves]
        
        # Sort by probability and take top N
        legal_moves.sort(key=lambda x: x[1], reverse=True)
        return legal_moves[:n]


# ============================================================================
# VALUE MODEL (18 channels, 12 residual blocks, GroupNorm + tanh)
# ============================================================================
class ValueModel(nn.Module):
    """
    Value network using GroupNorm (stable on huge datasets)
    Trained to output values in [-1, 1], later scaled to centipawns.
    """

    def __init__(self, channels=128, num_blocks=12):
        super().__init__()

        # Input conv
        self.conv_in = nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=True)
        self.gn_in = nn.GroupNorm(num_groups=32, num_channels=channels)

        # Residual tower using ResidualBlockGN
        self.blocks = nn.Sequential(
            *[ResidualBlockGN(channels, use_bias=True) for _ in range(num_blocks)]
        )

        # Value head
        self.conv_value = nn.Conv2d(channels, 8, kernel_size=1, bias=True)
        self.gn_value = nn.GroupNorm(num_groups=8, num_channels=8)

        self.fc_value1 = nn.Linear(8 * 8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv + GN + ReLU
        x = F.relu(self.gn_in(self.conv_in(x)))
        x = self.blocks(x)

        # Value head
        v = F.relu(self.gn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = self.fc_value2(v)
        return v

    @torch.no_grad()
    def evaluate_position(self, board: chess.Board) -> float:
        TARGET_SCALE = 2000.0  # same as training
        self.eval()

        x = ValueBoardEncoder.encode_board(board).unsqueeze(0)
        x = x.to(next(self.parameters()).device)

        v_scaled = self.forward(x).squeeze().item()
        return v_scaled * TARGET_SCALE

# ============================================================================
# MINIMAX SEARCH ENGINE
# ============================================================================
class MinimaxSearch:
    """
    Minimax search with alpha-beta pruning using policy + value models
    """
    
    def __init__(self, policy_model: PolicyModel, value_model: ValueModel):
        self.policy_model = policy_model
        self.value_model = value_model
        self.nodes_searched = 0
    
    def search(self, board: chess.Board, depth: int = 3, top_n: int = 10) -> Tuple[chess.Move, float]:
        """
        Minimax search to find best move
        
        Args:
            board: Current chess position
            depth: Search depth (ply)
            top_n: Number of candidate moves from policy at each node
        
        Returns:
            (best_move, evaluation) tuple
        """
        self.nodes_searched = 0
        
        # Get candidate moves from policy
        candidates = self.policy_model.get_top_n_moves(board, n=top_n)
        
        if not candidates:
            # Fallback
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return legal_moves[0], 0.0
            else:
                raise ValueError("No legal moves available")
        
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Search each candidate
        for move, policy_prob in candidates:
            board.push(move)
            
            # Recursive minimax (negate for opponent)
            score = -self._minimax(board, depth - 1, -beta, -alpha, top_n)
            
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_move, best_score
    
    def _minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, top_n: int) -> float:
        """Recursive minimax with alpha-beta pruning"""
        self.nodes_searched += 1
        
        # Base case
        if depth == 0 or board.is_game_over():
            if board.is_checkmate():
                return -10000  # Loss
            elif board.is_stalemate() or board.is_insufficient_material():
                return 0  # Draw
            else:
                return self.value_model.evaluate_position(board)
        
        # Get candidate moves
        n_moves = max(5, top_n - (3 - depth))
        candidates = self.policy_model.get_top_n_moves(board, n=n_moves)
        
        if not candidates:
            if board.is_checkmate():
                return -10000
            else:
                return 0
        
        max_eval = float('-inf')
        
        for move, _ in candidates:
            board.push(move)
            eval_score = -self._minimax(board, depth - 1, -beta, -alpha, top_n)
            board.pop()
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            if beta <= alpha:
                break  # Pruning
        
        return max_eval


# ============================================================================
# MAIN CHESS BOT
# ============================================================================
class ChessBot:
    """Complete chess bot combining policy and value models with minimax search"""
    
    def __init__(self, policy_path: str = None, value_path: str = None):
        """
        Initialize bot with trained models
        
        Args:
            policy_path: Path to policy model weights
            value_path: Path to value model weights
        """
        # Initialize models
        self.policy_model = PolicyModel(policy_size=4672)
        self.value_model = ValueModel(channels=128, num_blocks=12)
        
        # Load trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        repo_id = "Kahanesque/chesshacks-anderrsens-bot"

        print("[INFO] Downloading weights from HuggingFace Hub...")

        policy_path = hf_hub_download(repo_id=repo_id, filename="policy_resnet.pt")
        value_path = hf_hub_download(repo_id=repo_id, filename="value_model_lichess.pth")

        # Load state dicts
        self.policy_model.load_state_dict(
            torch.load(policy_path, map_location=device, weights_only=False)
        )
        self.value_model.load_state_dict(
            torch.load(value_path, map_location=device, weights_only=False)
        )

        print("[OK] Models loaded from HuggingFace Hub.")
        # Set models to evaluation mode
        self.policy_model.eval()
        self.value_model.eval()
        
        # Move to device
        self.policy_model.to(device)
        self.value_model.to(device)
        
        # Initialize search engine
        self.search = MinimaxSearch(self.policy_model, self.value_model)
    
    def get_move(self, board: chess.Board, depth: int = 3) -> chess.Move:
        """
        Get the bot's move for a given position
        
        Args:
            board: Current chess position
            depth: Search depth (default 3 ply)
        
        Returns:
            Best move selected by minimax search
        """
        with torch.no_grad():
            move, eval_score = self.search.search(board, depth=depth, top_n=10)
            print(f"Evaluation: {eval_score:.1f} cp, Nodes: {self.search.nodes_searched}")
        
        return move


# ============================================================================
# CHESSHACKS PLATFORM INTERFACE
# ============================================================================

from .utils import chess_manager, GameContext
from chess import Move

# ============================================================================
# INITIALIZATION (runs once when bot starts)
# ============================================================================

# Initialize bot with trained models
bot = ChessBot()
print("[OK] Chess bot initialized with minimax search")


# ============================================================================
# ENTRYPOINT (called every time bot needs to make a move)
# ============================================================================

@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    """
    ChessHacks platform entrypoint
    
    Args:
        ctx: GameContext containing current board state
    
    Returns:
        python-chess Move object (legal move for current position)
    """
    board = ctx.board
    
    # Ensure there are legal moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Get move from bot using minimax search
    try:
        move = bot.get_move(board, depth=3)
        
        # Log move probabilities from policy model
        # IMPORTANT: Keys must be Move objects, not tuples!
        with torch.no_grad():
            candidates = bot.policy_model.get_top_n_moves(board, n=len(legal_moves))
            # Create dict with Move objects as keys (not tuples)
            move_probs = {move: prob for move, prob in candidates}
            ctx.logProbabilities(move_probs)
        
        return move
    
    except Exception as e:
        print(f"Error in bot.get_move: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return first legal move
        ctx.logProbabilities({})
        return legal_moves[0]


# ============================================================================
# RESET (called when a new game begins)
# ============================================================================

@chess_manager.reset
def reset_game(ctx: GameContext):
    """Called at the start of each new game"""
    bot.search.nodes_searched = 0