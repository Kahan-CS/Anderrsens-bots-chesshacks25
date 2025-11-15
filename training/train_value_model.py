"""
Value Model Training Script

Trains a chess position evaluation model using the HuggingFace chess-evaluations dataset
Dataset: https://huggingface.co/datasets/ssingh22/chess-evaluations
Subset: evals_large (13M positions with Stockfish depth 22 evaluations)

Usage:
    python train_value_model.py --epochs 10 --batch_size 256 --lr 0.001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import chess
import argparse
from tqdm import tqdm
import os


# ============================================================================
# BOARD ENCODING (must match main.py exactly)
# ============================================================================
class BoardEncoder:
    """17-channel board encoding matching main.py"""
    
    @staticmethod
    def encode_board(board: chess.Board) -> torch.Tensor:
        """Convert board to tensor (17, 8, 8)"""
        planes = torch.zeros(17, 8, 8, dtype=torch.float32)
        
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


# ============================================================================
# MODEL ARCHITECTURE (must match main.py exactly)
# ============================================================================
class ResidualBlock(nn.Module):
    """Residual block"""
    
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
        return F.relu(residual + x)


class ValueModel(nn.Module):
    """Value model - same as main.py"""
    
    def __init__(self, channels=128, num_blocks=12):
        super().__init__()
        
        self.conv_in = nn.Conv2d(17, channels, kernel_size=3, padding=1)
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


# ============================================================================
# DATASET
# ============================================================================
class ChessEvaluationDataset(Dataset):
    """
    Dataset for chess position evaluations
    
    Loads from HuggingFace dataset: ssingh22/chess-evaluations (evals_large subset)
    """
    
    def __init__(self, split='train', max_samples=None, cache_dir=None):
        """
        Args:
            split: 'train' or 'test' (we'll create test split from train)
            max_samples: Limit dataset size for faster training/testing
            cache_dir: Directory to cache downloaded dataset
        """
        print(f"Loading dataset (split={split})...")
        
        # Load from HuggingFace
        dataset = load_dataset(
            "ssingh22/chess-evaluations",
            "evals_large",  # 13M positions subset
            split="train",
            cache_dir=cache_dir
        )
        
        # Create train/test split (90/10)
        if split == 'train':
            dataset = dataset.select(range(0, int(len(dataset) * 0.9)))
        else:  # test
            dataset = dataset.select(range(int(len(dataset) * 0.9), len(dataset)))
        
        # Limit size if requested
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = dataset
        print(f"Loaded {len(self.data)} positions")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            board_tensor: (17, 8, 8) tensor
            eval_cp: Evaluation in centipawns (float)
        """
        item = self.data[idx]
        fen = item['FEN']
        eval_str = item['Evaluation']
        
        # Parse FEN
        try:
            board = chess.Board(fen)
        except:
            # Invalid FEN, return zeros
            return torch.zeros(17, 8, 8), torch.tensor(0.0)
        
        # Parse evaluation
        eval_cp = self._parse_evaluation(eval_str, board.turn)
        
        # Encode board
        board_tensor = BoardEncoder.encode_board(board)
        
        return board_tensor, torch.tensor(eval_cp, dtype=torch.float32)
    
    def _parse_evaluation(self, eval_str: str, turn: chess.Color) -> float:
        """
        Parse evaluation string to centipawns from current player's perspective
        
        Format examples:
        - "+56" = white is up 56 centipawns
        - "-10" = black is up 10 centipawns
        - "#+6" = white has mate in 6
        - "#-3" = black has mate in 3
        
        Returns: Evaluation from perspective of player to move
        """
        eval_str = eval_str.strip()
        
        # Handle mate scores
        if '#' in eval_str:
            # Mate in N moves
            mate_in = int(eval_str.replace('#', '').replace('+', '').replace('-', ''))
            mate_score = 10000 - abs(mate_in) * 10  # Prefer faster mates
            
            # Determine who is winning
            if '+' in eval_str or (eval_str.startswith('#') and not '-' in eval_str):
                # White is winning
                eval_white = mate_score
            else:
                # Black is winning
                eval_white = -mate_score
        else:
            # Regular centipawn evaluation
            try:
                eval_white = float(eval_str)
            except:
                eval_white = 0.0
        
        # Convert to current player's perspective
        if turn == chess.WHITE:
            return eval_white
        else:
            return -eval_white


# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for board_tensors, eval_targets in progress_bar:
        board_tensors = board_tensors.to(device)
        eval_targets = eval_targets.to(device).unsqueeze(1)  # (batch, 1)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(board_tensors)
        
        # Compute loss
        loss = criterion(predictions, eval_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for board_tensors, eval_targets in tqdm(dataloader, desc="Evaluating"):
            board_tensors = board_tensors.to(device)
            eval_targets = eval_targets.to(device).unsqueeze(1)
            
            predictions = model(board_tensors)
            loss = criterion(predictions, eval_targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_value_model(
    epochs=10,
    batch_size=256,
    lr=0.001,
    channels=128,
    num_blocks=12,
    max_train_samples=None,
    max_test_samples=10000,
    save_path="value_model.pth",
    device=None
):
    """
    Main training function
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        channels: Model width
        num_blocks: Number of residual blocks
        max_train_samples: Limit training set size (for testing)
        max_test_samples: Limit test set size
        save_path: Where to save trained model
        device: torch device (None = auto-detect)
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("\n" + "="*50)
    print("LOADING TRAINING DATA")
    print("="*50)
    train_dataset = ChessEvaluationDataset(
        split='train',
        max_samples=max_train_samples
    )
    
    print("\n" + "="*50)
    print("LOADING TEST DATA")
    print("="*50)
    test_dataset = ChessEvaluationDataset(
        split='test',
        max_samples=max_test_samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = ValueModel(channels=channels, num_blocks=num_blocks)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # Mean squared error for centipawn prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved best model to {save_path}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Model saved to: {save_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train chess value model')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    # Model architecture
    parser.add_argument('--channels', type=int, default=128,
                        help='Number of channels in residual blocks')
    parser.add_argument('--num_blocks', type=int, default=12,
                        help='Number of residual blocks')
    
    # Dataset
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit training samples (for testing)')
    parser.add_argument('--max_test_samples', type=int, default=10000,
                        help='Limit test samples')
    
    # Output
    parser.add_argument('--save_path', type=str, default='value_model.pth',
                        help='Where to save trained model')
    
    args = parser.parse_args()
    
    # Run training
    train_value_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        channels=args.channels,
        num_blocks=args.num_blocks,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()