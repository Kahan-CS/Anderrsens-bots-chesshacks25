"""
Value Model Training on Modal (Serverless GPU)

Trains chess value model on Modal's cloud GPUs with:
- Automatic checkpoint saving/resuming
- Progress tracking (samples trained)
- Cost-efficient GPU selection
- Remote training with local results

Setup:
    1. pip install modal
    2. modal setup (authenticate)
    3. Run: modal run train_value_modal.py::main

Usage Examples:
    # Test run (5 min, ~$0.10)
    modal run train_value_modal.py::main --epochs 2 --max-samples 10000

    # Initial training (1 hr, ~$1.10)
    modal run train_value_modal.py::main --epochs 10 --max-samples 100000

    # Resume training
    modal run train_value_modal.py::main --resume --epochs 20

    # Download trained model
    modal run train_value_modal.py::download

    # List all saved models
    modal run train_value_modal.py::list_models

Cost with A10G GPU (~$1.10/hr):
    - 10k samples, 2 epochs: ~5 min = $0.10
    - 100k samples, 10 epochs: ~1 hr = $1.10
    - 1M samples, 10 epochs: ~10 hr = $11

Note: No .env file or secrets needed - Modal handles authentication automatically
"""

import modal
from pathlib import Path

# ============================================================================
# MODAL APP CONFIGURATION
# ============================================================================

# Create Modal app
app = modal.App("chess-value-training")

# GPU selection (updated to new string format)
# A10G: $1.10/hr, 24GB VRAM - RECOMMENDED for your budget
# With fallback to other GPUs if A10G unavailable
GPU_CONFIG = ["A10G", "L4", "T4"]  # Try A10G first, fallback to L4/T4

# Create persistent volume for checkpoints and models
volume = modal.Volume.from_name("chess-models", create_if_missing=True)
VOLUME_PATH = "/vol"

# Docker image with all dependencies (fixed versions for compatibility)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",  # Latest stable torch
        "datasets==3.2.0",  # HuggingFace datasets (doesn't need transformers)
        "python-chess==1.999",
        "tqdm==4.67.1",
        "numpy<2.0",  # Pin numpy to 1.x for compatibility
        "huggingface-hub==0.26.5",
    )
)


# ============================================================================
# TRAINING CODE (runs on Modal GPU)
# ============================================================================

@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 4,  # 4 hour timeout
)
def train_on_modal(
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.001,
    max_samples: int = 100000,
    channels: int = 128,
    num_blocks: int = 12,
    resume: bool = False,
):
    """
    Main training function that runs on Modal GPU
    
    This function contains all the training logic and will execute remotely.
    All imports must be inside the function for Modal to work correctly.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from datasets import load_dataset
    import chess
    from tqdm import tqdm
    import json
    from pathlib import Path
    import sys
    
    print("=" * 70)
    print("CHESS VALUE MODEL TRAINING ON MODAL")
    print("=" * 70)
    
    # Verify GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ WARNING: No GPU detected! Training will be very slow.")
    
    print(f"✓ Training samples: {max_samples:,}")
    print(f"✓ Epochs: {epochs}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Learning rate: {lr}")
    print(f"✓ Resume: {resume}")
    print("=" * 70)
    
    # ========================================================================
    # BOARD ENCODING (Fixed: 18 channels for proper castling encoding)
    # ========================================================================
    class BoardEncoder:
        @staticmethod
        def encode_board(board):
            # Fixed: Use 18 channels to properly encode all castling rights
            planes = torch.zeros(18, 8, 8, dtype=torch.float32)
            
            piece_idx = {
                chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
            }
            
            # Channels 0-11: Piece positions
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
            
            # Channel 13: Turn (1 if white to move)
            planes[13, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
            
            # Channels 14-17: Castling rights (properly encoded)
            planes[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
            planes[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
            planes[16, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
            planes[17, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
            
            return planes
    
    # ========================================================================
    # MODEL ARCHITECTURE
    # ========================================================================
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
            return F.relu(residual + x)
    
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
    
    # ========================================================================
    # DATASET
    # ========================================================================
    class ChessEvaluationDataset(Dataset):
        def __init__(self, split='train', max_samples=None):
            print(f"\nLoading dataset (split={split})...")
            
            try:
                dataset = load_dataset(
                    "ssingh22/chess-evaluations",
                    "evals_large",
                    split="train",
                    streaming=False
                )
                print(f"✓ Dataset loaded: {len(dataset):,} total positions")
            except Exception as e:
                print(f"✗ Error loading dataset: {e}")
                raise
            
            # Train/test split (90/10)
            split_idx = int(len(dataset) * 0.9)
            if split == 'train':
                dataset = dataset.select(range(0, split_idx))
            else:
                dataset = dataset.select(range(split_idx, len(dataset)))
            
            if max_samples and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
            
            self.data = dataset
            print(f"✓ Using {len(self.data):,} positions for {split}")
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            fen = item['FEN']
            eval_str = item['Evaluation']
            
            try:
                board = chess.Board(fen)
            except Exception:
                return torch.zeros(18, 8, 8), torch.tensor(0.0)
            
            eval_cp = self._parse_evaluation(eval_str, board.turn)
            board_tensor = BoardEncoder.encode_board(board)
            
            return board_tensor, torch.tensor(eval_cp, dtype=torch.float32)
        
        def _parse_evaluation(self, eval_str, turn):
            eval_str = eval_str.strip()
            
            if '#' in eval_str:
                # Mate score
                mate_in = int(eval_str.replace('#', '').replace('+', '').replace('-', ''))
                mate_score = 10000 - abs(mate_in) * 10
                if '+' in eval_str or (eval_str.startswith('#') and '-' not in eval_str):
                    eval_white = mate_score
                else:
                    eval_white = -mate_score
            else:
                try:
                    eval_white = float(eval_str)
                except Exception:
                    eval_white = 0.0
            
            return eval_white if turn == chess.WHITE else -eval_white
    
    # ========================================================================
    # CHECKPOINT MANAGEMENT
    # ========================================================================
    def save_checkpoint(model, optimizer, epoch, samples_trained, metrics, path):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'samples_trained': samples_trained,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path.name}")
    
    def load_checkpoint(model, optimizer, path):
        """Load training checkpoint if exists"""
        if Path(path).exists():
            print(f"\nLoading checkpoint: {path.name}")
            try:
                checkpoint = torch.load(path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                samples = checkpoint['samples_trained']
                metrics = checkpoint['metrics']
                print(f"✓ Resumed from epoch {epoch}, {samples:,} samples trained")
                return epoch, samples, metrics
            except Exception as e:
                print(f"✗ Error loading checkpoint: {e}")
                print("Starting fresh training...")
        
        return 0, 0, {'train_losses': [], 'test_losses': [], 'best_test_loss': float('inf')}
    
    # ========================================================================
    # TRAINING FUNCTIONS
    # ========================================================================
    def train_epoch(model, dataloader, optimizer, criterion, device, epoch, start_samples):
        model.train()
        total_loss = 0
        num_batches = 0
        samples_processed = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", file=sys.stdout)
        
        for board_tensors, eval_targets in progress_bar:
            try:
                board_tensors = board_tensors.to(device)
                eval_targets = eval_targets.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                predictions = model(board_tensors)
                loss = criterion(predictions, eval_targets)
                
                # Check for NaN
                if torch.isnan(loss):
                    print("⚠ NaN loss detected, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                samples_processed += len(board_tensors)
                
                progress_bar.set_postfix({
                    'loss': f"{total_loss / num_batches:.2f}",
                    'total_samples': f"{start_samples + samples_processed:,}"
                })
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠ OOM Error! Try reducing batch size.")
                    torch.cuda.empty_cache()
                    raise
                else:
                    raise
        
        return total_loss / max(num_batches, 1), samples_processed
    
    def evaluate(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for board_tensors, eval_targets in tqdm(dataloader, desc="Evaluating", file=sys.stdout):
                board_tensors = board_tensors.to(device)
                eval_targets = eval_targets.to(device).unsqueeze(1)
                predictions = model(board_tensors)
                loss = criterion(predictions, eval_targets)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    try:
        train_dataset = ChessEvaluationDataset(split='train', max_samples=max_samples)
        test_dataset = ChessEvaluationDataset(split='test', max_samples=min(10000, max_samples // 10))
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        raise
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )
    
    # Initialize model
    print(f"\nInitializing model (channels={channels}, blocks={num_blocks})...")
    model = ValueModel(channels=channels, num_blocks=num_blocks).to(device)
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Setup checkpoint paths
    checkpoint_dir = Path(VOLUME_PATH) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_file = checkpoint_dir / "value_model_checkpoint.pt"
    
    # Load checkpoint if resuming
    if resume:
        start_epoch, samples_trained, metrics = load_checkpoint(
            model, optimizer, checkpoint_file
        )
    else:
        start_epoch, samples_trained, metrics = 0, 0, {
            'train_losses': [], 'test_losses': [], 'best_test_loss': float('inf')
        }
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    for epoch in range(start_epoch, epochs):
        print(f"\n📊 Epoch {epoch + 1}/{epochs}")
        print("-" * 70)
        
        # Train
        try:
            train_loss, epoch_samples = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch + 1, samples_trained
            )
            samples_trained += epoch_samples
            metrics['train_losses'].append(train_loss)
            print(f"✓ Train Loss: {train_loss:.2f} cp²")
            print(f"✓ Total samples trained: {samples_trained:,}")
        except Exception as e:
            print(f"✗ Training error: {e}")
            raise
        
        # Evaluate
        try:
            test_loss = evaluate(model, test_loader, criterion, device)
            metrics['test_losses'].append(test_loss)
            print(f"✓ Test Loss: {test_loss:.2f} cp²")
        except Exception as e:
            print(f"✗ Evaluation error: {e}")
            raise
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Save checkpoint every epoch
        save_checkpoint(
            model, optimizer, epoch + 1, samples_trained, metrics, checkpoint_file
        )
        
        # Save best model
        if test_loss < metrics['best_test_loss']:
            metrics['best_test_loss'] = test_loss
            best_model_path = checkpoint_dir / "value_model_best.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best model! Test loss: {test_loss:.2f} cp²")
        
        # Commit volume changes
        volume.commit()
    
    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    final_model_path = checkpoint_dir / "value_model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # Save training summary
    summary = {
        'total_samples_trained': samples_trained,
        'total_epochs': epochs,
        'best_test_loss': metrics['best_test_loss'],
        'final_train_loss': metrics['train_losses'][-1] if metrics['train_losses'] else 0,
        'final_test_loss': metrics['test_losses'][-1] if metrics['test_losses'] else 0,
        'architecture': {
            'channels': channels,
            'num_blocks': num_blocks,
        }
    }
    
    summary_path = checkpoint_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    volume.commit()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"📊 Total samples trained: {samples_trained:,}")
    print(f"🏆 Best test loss: {metrics['best_test_loss']:.2f} cp²")
    print(f"💾 Best model: {best_model_path.name}")
    print(f"💾 Final model: {final_model_path.name}")
    print(f"📄 Summary: {summary_path.name}")
    print("=" * 70)
    
    return summary


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main(
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 0.001,
    max_samples: int = 100000,
    channels: int = 128,
    num_blocks: int = 12,
    resume: bool = False,
):
    """
    Run training on Modal GPU
    
    Usage:
        modal run train_value_modal.py::main
        modal run train_value_modal.py::main --epochs 5 --max-samples 50000
        modal run train_value_modal.py::main --resume  # Continue from checkpoint
    
    Args:
        epochs: Number of epochs to train
        batch_size: Batch size (reduce if OOM)
        lr: Learning rate
        max_samples: Max training samples (use 100k initially)
        channels: Model width
        num_blocks: Number of residual blocks
        resume: Resume from checkpoint
    """
    print("\n🚀 Starting training on Modal GPU...")
    print(f"Configuration:")
    print(f"  • Epochs: {epochs}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Max samples: {max_samples:,}")
    print(f"  • Learning rate: {lr}")
    print(f"  • Resume: {resume}")
    print()
    
    # Run training on Modal
    summary = train_on_modal.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_samples=max_samples,
        channels=channels,
        num_blocks=num_blocks,
        resume=resume,
    )
    
    print("\n✅ Training complete!")
    print("\n📊 Training Summary:")
    print(f"  • Samples trained: {summary['total_samples_trained']:,}")
    print(f"  • Best test loss: {summary['best_test_loss']:.2f} cp²")
    print(f"  • Final test loss: {summary['final_test_loss']:.2f} cp²")
    print("\n📦 Models saved to Modal volume 'chess-models'")
    print("\n📥 To download trained model:")
    print("  modal run train_value_modal.py::download")


# ============================================================================
# UTILITY: DOWNLOAD MODEL
# ============================================================================

@app.function(volumes={VOLUME_PATH: volume})
def download_model_data(remote_path: str):
    """Download model from Modal volume"""
    remote_file = Path(VOLUME_PATH) / remote_path
    if remote_file.exists():
        with open(remote_file, 'rb') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Model not found: {remote_path}")


@app.local_entrypoint()
def download(
    remote_path: str = "checkpoints/value_model_best.pth",
    local_path: str = "value_model.pth"
):
    """
    Download trained model from Modal
    
    Usage:
        modal run train_value_modal.py::download
        modal run train_value_modal.py::download --remote-path checkpoints/value_model_final.pth
    """
    print(f"📥 Downloading {remote_path} from Modal...")
    
    try:
        data = download_model_data.remote(remote_path)
        
        with open(local_path, 'wb') as f:
            f.write(data)
        
        print(f"✅ Model downloaded to {local_path}")
        print(f"   Size: {len(data) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\n💡 Try listing models first:")
        print("   modal run train_value_modal.py::list_models")


# ============================================================================
# UTILITY: LIST CHECKPOINTS
# ============================================================================

@app.function(volumes={VOLUME_PATH: volume})
def list_checkpoints_data():
    """List all saved checkpoints and models"""
    checkpoint_dir = Path(VOLUME_PATH) / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    
    files = []
    for f in checkpoint_dir.iterdir():
        stat = f.stat()
        files.append({
            'name': f.name,
            'size_mb': stat.st_size / 1024 / 1024,
            'modified': stat.st_mtime,
        })
    return files


@app.local_entrypoint()
def list_models():
    """
    List all saved models on Modal
    
    Usage:
        modal run train_value_modal.py::list_models
    """
    print("📋 Saved models on Modal:")
    files = list_checkpoints_data.remote()
    
    if not files:
        print("  (no models found)")
        print("\n💡 Train a model first:")
        print("   modal run train_value_modal.py::main")
    else:
        print()
        for f in sorted(files, key=lambda x: x['modified'], reverse=True):
            print(f"  • {f['name']:<40} {f['size_mb']:>8.2f} MB")