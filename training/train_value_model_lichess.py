# train_value_modal.py
"""
Train ValueModel on Lichess position evaluations (streaming).

Key features:
- Uses Lichess/chess-position-evaluations dataset (streaming)
- Random sampling ~25M positions (Bernoulli sampling)
- GroupNorm instead of BatchNorm for stability with streaming
- Clip cp to ±2000, scale to [-1,1] (TARGET_SCALE = 2000)
- HuberLoss(delta=0.2) in scaled domain
- Checkpointing to Modal volume "chess-models" (same names as before)
"""

import modal
from pathlib import Path

# ---------------- Modal configuration (unchanged naming)
app = modal.App("chess-value-training-lichess")
GPU_CONFIG = ["A10G", "L4", "T4"]
volume = modal.Volume.from_name("chess-models", create_if_missing=True)
VOLUME_PATH = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "datasets==3.12.0",
        "python-chess==1.999",
        "tqdm==4.67.1",
        "numpy<2.0",
        "huggingface-hub==0.26.5",
    )
)

# -----------------------------------------------------------------------------
# TRAIN FUNCTION (runs on Modal GPU)
# -----------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 8,
)
def train_on_modal(
    epochs: int = 2,
    batch_size: int = 256,
    lr: float = 3e-4,
    max_samples: int = 25_000_000,
    channels: int = 128,
    num_blocks: int = 12,
    resume: bool = False,
):
    """
    Trains the value model on Lichess streaming dataset.

    Important:
      - max_samples is the number of positions to sample for training (per *total* run)
      - sampling is Bernoulli-style to produce approx max_samples from the large pool
    """
    # local imports (Modal requirement)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import IterableDataset, DataLoader
    from datasets import load_dataset
    import chess
    from tqdm import tqdm
    import json, math, sys, random, time
    from pathlib import Path
    import numpy as np

    print("=" * 72)
    print("TRAIN VALUE MODEL - LICHESS (streaming)".center(72))
    print("=" * 72)
    print(f"Samples requested: {max_samples:,}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"Channels: {channels}, Residual blocks: {num_blocks}")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Scaling and clipping constants
    # -------------------------------------------------------------------------
    CLIP_CP = 2000.0
    TARGET_SCALE = CLIP_CP  # divide cp by TARGET_SCALE --> target in [-1,1]

    # -------------------------------------------------------------------------
    # Model architecture (GroupNorm used instead of BatchNorm)
    # -------------------------------------------------------------------------
    class ResidualBlockGN(nn.Module):
        def __init__(self, channels=128, num_groups=8, use_bias=True):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=use_bias)
            self.gn1 = nn.GroupNorm(num_groups, channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=use_bias)
            self.gn2 = nn.GroupNorm(num_groups, channels)

        def forward(self, x):
            identity = x
            out = F.relu(self.gn1(self.conv1(x)))
            out = self.gn2(self.conv2(out))
            return F.relu(out + identity)

    class ValueModel(nn.Module):
        def __init__(self, channels=128, num_blocks=12, num_groups=8):
            super().__init__()
            self.conv_in = nn.Conv2d(18, channels, kernel_size=3, padding=1)
            self.gn_in = nn.GroupNorm(num_groups, channels)
            self.blocks = nn.Sequential(
                *[ResidualBlockGN(channels, num_groups=num_groups, use_bias=True) for _ in range(num_blocks)]
            )
            self.conv_value = nn.Conv2d(channels, 8, kernel_size=1)
            self.gn_value = nn.GroupNorm(2, 8)  # group size 2 for small channel head
            self.fc_value1 = nn.Linear(8 * 8 * 8, 256)
            self.fc_value2 = nn.Linear(256, 1)

        def forward(self, x):
            x = F.relu(self.gn_in(self.conv_in(x)))
            x = self.blocks(x)
            v = F.relu(self.gn_value(self.conv_value(x)))
            v = v.view(v.size(0), -1)
            v = F.relu(self.fc_value1(v))
            v = self.fc_value2(v)
            return v

        @torch.no_grad()
        def evaluate_position(self, board: chess.Board) -> float:
            """
            Evaluate position (returns centipawns).
            Model outputs scaled value in [-1,1] -> multiply by TARGET_SCALE.
            """
            self.eval()
            board_tensor = encode_board_18(board).unsqueeze(0)
            device = next(self.parameters()).device
            board_tensor = board_tensor.to(device)
            scaled = self.forward(board_tensor).squeeze().item()
            return scaled * TARGET_SCALE

    # -------------------------------------------------------------------------
    # Board encoding (18 channels)
    # -------------------------------------------------------------------------
    def encode_board_18(board: chess.Board):
        t = torch.zeros(18, 8, 8, dtype=torch.float32)
        piece_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
        for s in chess.SQUARES:
            p = board.piece_at(s)
            if p:
                r = chess.square_rank(s)
                f = chess.square_file(s)
                idx = piece_idx[p.piece_type] + (6 if p.color == chess.BLACK else 0)
                t[idx, r, f] = 1.0
        t[12, :, :] = 0.0
        t[13, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        t[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        t[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        t[16, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        t[17, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        return t

    # -------------------------------------------------------------------------
    # Iterable dataset that streams and randomly samples rows until target reached
    # -------------------------------------------------------------------------
    class LichessStreamDataset(IterableDataset):
        def __init__(self, target_samples: int, split='train', seed: int = 42):
            self.target_samples = int(target_samples)
            self.split = split
            self.seed = seed
            # Total examples (approx) in dataset: used to compute sampling prob
            # Using conservative estimate from dataset card:
            self.TOTAL_EXAMPLES = 784_537_782
            self.sample_prob = float(self.target_samples) / float(self.TOTAL_EXAMPLES)
            if self.sample_prob <= 0.0 or self.sample_prob > 1.0:
                self.sample_prob = min(max(self.sample_prob, 0.000001), 1.0)
            # For reproducibility
            random.seed(self.seed)

        def __iter__(self):
            ds = load_dataset("Lichess/chess-position-evaluations", split=self.split, streaming=True)
            count = 0
            for item in ds:
                # Bernoulli sampling
                if random.random() < self.sample_prob:
                    # Parse fen and cp/mate
                    fen = item.get("fen") or item.get("FEN") or item.get("fenstr") or None
                    cp = item.get("cp")
                    mate = item.get("mate")
                    if fen is None:
                        continue
                    try:
                        board = chess.Board(fen)
                    except Exception:
                        continue

                    # Determine cp target
                    if cp is None:
                        # If mate provided, map to strong win/loss
                        if mate is not None:
                            cp_val = CLIP_CP if mate > 0 else -CLIP_CP
                        else:
                            # fallback: skip if no cp and no mate
                            continue
                    else:
                        cp_val = float(cp)

                    # Clip + scale
                    cp_val = max(-CLIP_CP, min(CLIP_CP, cp_val))
                    scaled = cp_val / TARGET_SCALE  # in [-1,1]

                    board_tensor = encode_board_18(board)
                    count += 1
                    yield board_tensor, torch.tensor(scaled, dtype=torch.float32)

                    if count >= self.target_samples:
                        break
            # End iteration

    # -------------------------------------------------------------------------
    # Checkpoint utilities (same names as earlier, stored in Modal volume)
    # -------------------------------------------------------------------------
    def save_checkpoint(model, optimizer, epoch, samples_trained, metrics, path):
        checkpoint = {
            'epoch': epoch,
            'samples_trained': samples_trained,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, path)
        print(f"[]Checkpoint saved: {path.name}")

    def load_checkpoint(model, optimizer, path):
        if Path(path).exists():
            print(f"\nLoading checkpoint: {path.name}")
            try:
                cp = torch.load(path, map_location='cpu')
                if isinstance(cp, dict) and 'model_state_dict' in cp:
                    model.load_state_dict(cp['model_state_dict'])
                else:
                    # raw state_dict
                    model.load_state_dict(cp)
                optimizer.load_state_dict(cp.get('optimizer_state_dict', {}))
                epoch = cp.get('epoch', 0)
                samples = cp.get('samples_trained', 0)
                metrics = cp.get('metrics', {'train_losses': [], 'test_losses': [], 'best_test_loss': float('inf')})
                print(f"[]Resumed from epoch {epoch}, {samples:,} samples trained")
                return epoch, samples, metrics
            except Exception as e:
                print(f"[ERR]Error loading checkpoint: {e}")
                print("Starting fresh training...")
        return 0, 0, {'train_losses': [], 'test_losses': [], 'best_test_loss': float('inf')}

    # -------------------------------------------------------------------------
    # Training loops (train_epoch and evaluate)
    # -------------------------------------------------------------------------
    def train_epoch(model, dataloader, optimizer, criterion, device, epoch, start_samples, steps_per_epoch):
        model.train()
        total_loss = 0.0
        total_mae_cp = 0.0
        num_batches = 0
        samples_processed = 0

        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}", file=sys.stdout)
        it = iter(dataloader)
        for step in range(steps_per_epoch):
            try:
                board_tensors, eval_targets = next(it)
            except StopIteration:
                break
            board_tensors = board_tensors.to(device)
            eval_targets = eval_targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(board_tensors)
            loss = criterion(predictions, eval_targets)

            if torch.isnan(loss):
                print("⚠ NaN loss detected, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds_cp = (predictions * TARGET_SCALE).detach()
            targets_cp = (eval_targets * TARGET_SCALE).detach()
            batch_mae_cp = torch.mean(torch.abs(preds_cp - targets_cp)).item()

            total_loss += loss.item()
            total_mae_cp += batch_mae_cp
            num_batches += 1
            samples_processed += board_tensors.size(0)

            pbar.update(1)
            pbar.set_postfix({
                'huber_scaled': f"{total_loss / num_batches:.5f}",
                'mae_cp': f"{total_mae_cp / num_batches:.1f}",
                'samples': f"{start_samples + samples_processed:,}"
            })

        pbar.close()
        avg_loss = total_loss / max(num_batches, 1)
        avg_mae = total_mae_cp / max(num_batches, 1)
        return avg_loss, avg_mae, samples_processed

    def evaluate(model, dataloader, criterion, device, steps):
        model.eval()
        total_loss = 0.0
        total_mae_cp = 0.0
        num_batches = 0

        with torch.no_grad():
            it = iter(dataloader)
            for i in range(steps):
                try:
                    board_tensors, eval_targets = next(it)
                except StopIteration:
                    break
                board_tensors = board_tensors.to(device)
                eval_targets = eval_targets.to(device).unsqueeze(1)

                predictions = model(board_tensors)
                loss = criterion(predictions, eval_targets)

                preds_cp = predictions * TARGET_SCALE
                targets_cp = eval_targets * TARGET_SCALE
                batch_mae_cp = torch.mean(torch.abs(preds_cp - targets_cp)).item()

                total_loss += loss.item()
                total_mae_cp += batch_mae_cp
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_mae = total_mae_cp / max(num_batches, 1)
        return avg_loss, avg_mae

    # -------------------------------------------------------------------------
    # Prepare datasets / loaders
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For streaming we use an IterableDataset that yields exactly max_samples entries
    total_samples = int(max_samples)
    # We'll split samples across epochs roughly evenly:
    samples_per_epoch = math.ceil(total_samples / max(1, epochs))
    print(f"Samples per epoch (approx): {samples_per_epoch:,}")

    train_ds = LichessStreamDataset(target_samples=samples_per_epoch, split='train', seed=42)
    # small validation set: sample a separate small stream (1% of per-epoch or min 10k)
    val_samples = min(10000, max(1000, samples_per_epoch // 100))
    val_ds = LichessStreamDataset(target_samples=val_samples, split='train', seed=9999)

    # Convert iterable dataset to DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    # steps per epoch we will perform (to limit training loop)
    steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
    val_steps = math.ceil(val_samples / batch_size)

    # -------------------------------------------------------------------------
    # Initialize model / optimizer / criterion / scheduler
    # -------------------------------------------------------------------------
    print("\nInitializing model...")
    model = ValueModel(channels=channels, num_blocks=num_blocks, num_groups=8).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.HuberLoss(delta=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # -------------------------------------------------------------------------
    # Checkpoint paths (same names as before)
    # -------------------------------------------------------------------------
    checkpoint_dir = Path(VOLUME_PATH) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_file = checkpoint_dir / "value_model_checkpoint.pt"
    best_model_path = checkpoint_dir / "value_model_best.pth"
    final_model_path = checkpoint_dir / "value_model_final.pth"

    # resume?
    if resume:
        start_epoch, samples_trained, metrics = load_checkpoint(model, optimizer, checkpoint_file)
    else:
        start_epoch, samples_trained, metrics = 0, 0, {'train_losses': [], 'test_losses': [], 'best_test_loss': float('inf')}

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STARTING TRAINING")
    print("=" * 72)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        try:
            train_loss, train_mae_cp, processed = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch + 1, samples_trained, steps_per_epoch
            )
            samples_trained += processed
            metrics['train_losses'].append(train_loss)
            print(f"[]Train Loss (scaled Huber): {train_loss:.5f}")
            print(f"[]Train MAE: {train_mae_cp:.1f} centipawns")
        except Exception as e:
            print("[ERR]Training error:", e)
            raise

        try:
            test_loss, test_mae_cp = evaluate(model, val_loader, criterion, device, val_steps)
            metrics['test_losses'].append(test_loss)
            print(f"[]Val Loss (scaled Huber): {test_loss:.5f}")
            print(f"[]Val MAE: {test_mae_cp:.1f} centipawns")
        except Exception as e:
            print("[ERR]Evaluation error:", e)
            raise

        scheduler.step(test_loss)

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch + 1, samples_trained, metrics, checkpoint_file)

        # Save best model
        if test_loss < metrics['best_test_loss']:
            metrics['best_test_loss'] = test_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best model! Val loss: {test_loss:.5f} (scaled)")

        volume.commit()

    # Save final model weights
    torch.save(model.state_dict(), final_model_path)

    # Save training summary
    summary = {
        'total_samples_trained': samples_trained,
        'total_epochs': epochs,
        'best_test_loss_scaled': metrics['best_test_loss'],
        'final_train_loss_scaled': metrics['train_losses'][-1] if metrics['train_losses'] else 0,
        'final_test_loss_scaled': metrics['test_losses'][-1] if metrics['test_losses'] else 0,
        'architecture': {'channels': channels, 'num_blocks': num_blocks},
        'clip_cp': CLIP_CP,
        'target_scale': TARGET_SCALE
    }

    summary_path = checkpoint_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print("\n" + "=" * 72)
    print("✅ TRAINING COMPLETE")
    print("=" * 72)
    print(f"📊 Total samples trained: {samples_trained:,}")
    print(f"🏆 Best model: {best_model_path.name}")
    print(f"💾 Final model: {final_model_path.name}")
    print(f"📄 Summary: {summary_path.name}")
    print("=" * 72)

    return summary


# -----------------------------------------------------------------------------
# Local entrypoint (keeps the same CLI from earlier)
# -----------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    epochs: int = 2,
    batch_size: int = 256,
    lr: float = 3e-4,
    max_samples: int = 25_000_000,
    channels: int = 128,
    num_blocks: int = 12,
    resume: bool = False,
):
    print("\n Launching Lichess training (streaming)...")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Max samples: {max_samples:,}")
    print(f"- LR: {lr}")
    print(f"- Resume: {resume}")
    print()
    summary = train_on_modal.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_samples=max_samples,
        channels=channels,
        num_blocks=num_blocks,
        resume=resume,
    )
    print("\nTraining launched (remote).")
    return summary


# -----------------------------------------------------------------------------
# Download & List utilities (same as before)
# -----------------------------------------------------------------------------
@app.function(volumes={VOLUME_PATH: volume})
def download_model_data(remote_path: str):
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
    print(f"Downloading {remote_path} from Modal...")
    try:
        data = download_model_data.remote(remote_path)
        with open(local_path, 'wb') as f:
            f.write(data)
        print(f"✅ Model downloaded to {local_path}")
        print(f"   Size: {len(data) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nTry listing models first:")
        print(" modal run train_value_modal.py::list_models")


@app.function(volumes={VOLUME_PATH: volume})
def list_checkpoints_data():
    checkpoint_dir = Path(VOLUME_PATH) / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    files = []
    for f in checkpoint_dir.iterdir():
        stat = f.stat()
        files.append({'name': f.name, 'size_mb': stat.st_size / 1024 / 1024, 'modified': f.stat().st_mtime})
    return files


@app.local_entrypoint()
def list_models():
    print("Saved models on Modal:")
    files = list_checkpoints_data.remote()
    if not files:
        print("  (no models found)")
        print("\nTrain a model first:")
        print("   modal run train_value_modal.py::main")
    else:
        for f in sorted(files, key=lambda x: x['modified'], reverse=True):
            print(f"  • {f['name']:<40} {f['size_mb']:>8.2f} MB")
