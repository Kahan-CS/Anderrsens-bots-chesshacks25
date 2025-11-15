# main.py - Chess Bot Entry Point for ChessHacks Platform

"""
Chess Bot with Policy Model + Value Model + Search

Architecture:
1. Policy Model generates top-N candidate moves
2. Value Model evaluates board positions
3. Search function selects best move from candidates

Models are trained separately and loaded from weights/ directory
"""

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import os

# ============================================================================
# BOARD ENCODING
# ============================================================================
class BoardEncoder:
    """Encodes chess board into tensor representation for neural networks"""
    
    @staticmethod
    def encode_board(board: chess.Board) -> torch.Tensor:
        """
        Convert board to neural network input
        
        Encoding scheme (12 planes):
        - Planes 0-5: White pieces (P, N, B, R, Q, K)
        - Planes 6-11: Black pieces (P, N, B, R, Q, K)
        
        Returns: tensor of shape (12, 8, 8)
        """
        planes = torch.zeros(12, 8, 8, dtype=torch.float32)
        
        piece_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                plane_idx = piece_idx[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane_idx += 6
                
                planes[plane_idx, rank, file] = 1.0
        
        return planes
    
    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        """
        Convert move to action index for policy model
        Simple encoding: from_square * 64 + to_square
        (For promotion moves, you may need more sophisticated encoding)
        """
        return move.from_square * 64 + move.to_square
    
    @staticmethod
    def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
        """Convert action index back to move (if legal)"""
        from_square = index // 64
        to_square = index % 64
        move = chess.Move(from_square, to_square)
        
        if move in board.legal_moves:
            return move
        return None


# ============================================================================
# POLICY MODEL
# ============================================================================
class PolicyModel(nn.Module):
    """
    Neural network that predicts move probabilities
    
    Architecture: Convolutional layers + Policy head
    Input: (12, 8, 8) board representation
    Output: Probability distribution over moves
    """
    
    def __init__(self, input_channels: int = 12, hidden_channels: int = 128, num_res_blocks: int = 4):
        super().__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(hidden_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)  # 64*64 possible moves
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Board tensor of shape (batch, 12, 8, 8)
        
        Returns:
            Move logits of shape (batch, 4096)
        """
        # Input convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        return policy
    
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
        
        # Encode board
        board_tensor = BoardEncoder.encode_board(board).unsqueeze(0)  # Add batch dimension
        
        # Get move logits
        logits = self.forward(board_tensor).squeeze(0)  # Remove batch dimension
        probs = F.softmax(logits, dim=0)
        
        # Filter for legal moves and get top N
        legal_moves = []
        for move in board.legal_moves:
            move_idx = BoardEncoder.move_to_index(move)
            if move_idx < len(probs):
                legal_moves.append((move, probs[move_idx].item()))
        
        # Sort by probability and take top N
        legal_moves.sort(key=lambda x: x[1], reverse=True)
        return legal_moves[:n]


# ============================================================================
# VALUE MODEL
# ============================================================================
class ValueModel(nn.Module):
    """
    Neural network that evaluates board positions
    
    Architecture: Convolutional layers + Value head
    Input: (12, 8, 8) board representation
    Output: Position evaluation (win probability from white's perspective)
    """
    
    def __init__(self, input_channels: int = 12, hidden_channels: int = 128, num_res_blocks: int = 4):
        super().__init__()
        
        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(hidden_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_res_blocks)
        ])
        
        # Value head
        self.value_conv = nn.Conv2d(hidden_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Board tensor of shape (batch, 12, 8, 8)
        
        Returns:
            Value evaluation of shape (batch, 1) in range [-1, 1]
            (1 = white winning, -1 = black winning, 0 = draw)
        """
        # Input convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]
        
        return value
    
    @torch.no_grad()
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a single board position
        
        Args:
            board: Chess position to evaluate
        
        Returns:
            Evaluation score from current player's perspective
            Positive = current player is winning
            Negative = current player is losing
        """
        self.eval()
        
        # Encode board
        board_tensor = BoardEncoder.encode_board(board).unsqueeze(0)  # Add batch dimension
        
        # Get evaluation (from white's perspective)
        eval_white = self.forward(board_tensor).squeeze().item()
        
        # Flip perspective if black to move
        if board.turn == chess.BLACK:
            return -eval_white
        return eval_white


# ============================================================================
# RESIDUAL BLOCK (shared by both models)
# ============================================================================
class ResidualBlock(nn.Module):
    """Residual block for deep network"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


# ============================================================================
# SEARCH ENGINE
# ============================================================================
class SearchEngine:
    """
    Search algorithm using policy + value models
    
    Strategy: Get top-N moves from policy, evaluate resulting positions
    with value model, select best move.
    """
    
    def __init__(self, policy_model: PolicyModel, value_model: ValueModel):
        self.policy_model = policy_model
        self.value_model = value_model
    
    def search(self, board: chess.Board, top_n: int = 10, depth: int = 1) -> chess.Move:
        """
        Select best move using policy + value models
        
        Args:
            board: Current chess position
            top_n: Number of candidate moves to consider from policy
            depth: Search depth (1 = evaluate immediate positions only)
        
        Returns:
            Best move selected
        """
        # Get candidate moves from policy model
        candidates = self.policy_model.get_top_n_moves(board, n=top_n)
        
        if not candidates:
            # Fallback: return any legal move
            return list(board.legal_moves)[0]
        
        # Evaluate each candidate
        best_move = None
        best_score = float('-inf')
        
        for move, policy_prob in candidates:
            # Make the move
            board.push(move)
            
            # Evaluate the resulting position
            # Note: Value is from opponent's perspective after the move
            # so we negate it to get our perspective
            if depth == 1:
                score = -self.value_model.evaluate_position(board)
            else:
                # For depth > 1, implement recursive search
                score = self._search_recursive(board, depth - 1, alpha=float('-inf'), beta=float('inf'))
            
            # Undo the move
            board.pop()
            
            # Track best move
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _search_recursive(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """
        Recursive minimax search with alpha-beta pruning
        
        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
        
        Returns:
            Position evaluation from current player's perspective
        """
        # Base case: evaluate position
        if depth == 0 or board.is_game_over():
            return -self.value_model.evaluate_position(board)
        
        # Get candidate moves (fewer at deeper levels to save time)
        n_moves = max(5, 10 - depth)
        candidates = self.policy_model.get_top_n_moves(board, n=n_moves)
        
        max_eval = float('-inf')
        for move, _ in candidates:
            board.push(move)
            eval_score = -self._search_recursive(board, depth - 1, -beta, -alpha)
            board.pop()
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            if beta <= alpha:
                break  # Beta cutoff
        
        return max_eval


# ============================================================================
# MAIN CHESS BOT
# ============================================================================
class ChessBot:
    """
    Complete chess bot combining policy and value models
    
    Models are loaded from saved weights (trained separately)
    """
    
    def __init__(self, policy_path: str = None, value_path: str = None):
        """
        Initialize bot with trained models
        
        Args:
            policy_path: Path to policy model weights (.pth file)
            value_path: Path to value model weights (.pth file)
        """
        # Set default paths relative to this file
        if policy_path is None:
            policy_path = os.path.join(os.path.dirname(__file__), "weights", "policy_model.pth")
        if value_path is None:
            value_path = os.path.join(os.path.dirname(__file__), "weights", "value_model.pth")
        
        # Initialize models
        self.policy_model = PolicyModel()
        self.value_model = ValueModel()
        
        # Load trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if os.path.exists(policy_path):
            self.policy_model.load_state_dict(torch.load(policy_path, map_location=device))
            print(f"✓ Loaded policy model from {policy_path}")
        else:
            print(f"⚠ Warning: Policy model not found at {policy_path}, using random weights")
        
        if os.path.exists(value_path):
            self.value_model.load_state_dict(torch.load(value_path, map_location=device))
            print(f"✓ Loaded value model from {value_path}")
        else:
            print(f"⚠ Warning: Value model not found at {value_path}, using random weights")
        
        # Set models to evaluation mode
        self.policy_model.eval()
        self.value_model.eval()
        
        # Move to device
        self.policy_model.to(device)
        self.value_model.to(device)
        
        # Initialize search engine
        self.search_engine = SearchEngine(self.policy_model, self.value_model)
    
    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get the bot's move for a given position
        
        Args:
            board: Current chess position
        
        Returns:
            Best move selected by the bot
        """
        with torch.no_grad():
            move = self.search_engine.search(board, top_n=10, depth=1)
        
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
# Models will be loaded from weights/ directory
bot = ChessBot()
print("✓ Chess bot initialized")


# ============================================================================
# ENTRYPOINT (called every time bot needs to make a move)
# ============================================================================

@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    """
    ChessHacks platform entrypoint
    
    Args:
        ctx: GameContext containing current board state and utilities
    
    Returns:
        python-chess Move object (legal move for current position)
    """
    board = ctx.board
    
    # Ensure there are legal moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    # Get move from bot using policy + value search
    try:
        move = bot.get_move(board)
        
        # Log move probabilities from policy model for analysis
        # Get top candidates and their probabilities
        with torch.no_grad():
            candidates = bot.policy_model.get_top_n_moves(board, n=len(legal_moves))
            move_probs = {m: prob for m, prob in candidates}
            ctx.logProbabilities(move_probs)
        
        return move
    
    except Exception as e:
        print(f"Error in bot.get_move: {e}")
        # Fallback: return first legal move
        ctx.logProbabilities({})
        return legal_moves[0]


# ============================================================================
# RESET (called when a new game begins)
# ============================================================================

@chess_manager.reset
def reset_game(ctx: GameContext):
    """
    Called at the start of each new game
    
    Use this to:
    - Clear any caches
    - Reset model state
    - Clear search history
    """
    # Currently no stateful components to reset
    # Add here if you implement caching, transposition tables, etc.
    pass