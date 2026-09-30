"""
PharmaLens — Training Utilities

Generic training loop for PyTorch models with:
- Early stopping
- Learning rate scheduling
- Checkpoint saving
- Training history logging
- Loss curves generation
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("pharmalens.training")


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """
    Generic PyTorch model trainer for DTI prediction.

    Handles training loop, validation, early stopping, checkpointing,
    and history logging for both DeepDTA and GNN models.

    Args:
        model: PyTorch model.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: torch.device.
        scheduler: Optional LR scheduler.
        early_stopping_patience: Epochs to wait before stopping.
        checkpoint_dir: Directory for saving checkpoints.
        model_name: Name for logging and file naming.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        criterion=None,
        device=None,
        scheduler=None,
        early_stopping_patience: int = 20,
        checkpoint_dir: Optional[Path] = None,
        model_name: str = "model",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.MSELoss()
        self.device = device or torch.device("cpu")
        self.scheduler = scheduler
        self.model_name = model_name

        self.model.to(self.device)

        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Training history
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "epoch_time": [],
        }

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.total_training_time = 0

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            # Handle different batch formats (DeepDTA vs GNN)
            if len(batch) == 3:
                # DeepDTA: (drug, protein, score)
                drug, protein, scores = batch
                drug = drug.to(self.device)
                protein = protein.to(self.device)
                scores = scores.to(self.device)
                predictions = self.model(drug, protein)
            elif hasattr(batch, 'y'):
                # PyG batch (GNN)
                batch = batch.to(self.device)
                predictions = self.model(batch)
                scores = batch.y.squeeze()
            else:
                raise ValueError(f"Unexpected batch format: {type(batch)}")

            loss = self.criterion(predictions, scores)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> tuple:
        """Run validation and return loss + predictions."""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        all_preds = []
        all_true = []

        for batch in val_loader:
            if len(batch) == 3:
                drug, protein, scores = batch
                drug = drug.to(self.device)
                protein = protein.to(self.device)
                scores = scores.to(self.device)
                predictions = self.model(drug, protein)
            elif hasattr(batch, 'y'):
                batch = batch.to(self.device)
                predictions = self.model(batch)
                scores = batch.y.squeeze()
            else:
                raise ValueError(f"Unexpected batch format")

            loss = self.criterion(predictions, scores)
            total_loss += loss.item()
            n_batches += 1

            all_preds.append(predictions.cpu().numpy())
            all_true.append(scores.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)

        return avg_loss, all_preds, all_true

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
    ) -> dict:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum number of epochs.
            verbose: Print progress every epoch.

        Returns:
            dict: Training history.
        """
        logger.info(f"Starting training: {self.model_name} for {epochs} epochs")
        t_total_start = time.time()

        for epoch in range(1, epochs + 1):
            t_epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, _, _ = self.validate(val_loader)

            epoch_time = time.time() - t_epoch_start

            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch)

            # Logging
            if verbose and (epoch % 10 == 0 or epoch == 1):
                logger.info(
                    f"  Epoch {epoch:>4d}/{epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
                )
                print(
                    f"  Epoch {epoch:>4d}/{epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
                )

            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"  Early stopping at epoch {epoch} (patience={self.early_stopping.patience})")
                print(f"  Early stopping at epoch {epoch}")
                break

        self.total_training_time = time.time() - t_total_start

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        logger.info(
            f"Training complete: {epoch} epochs, "
            f"best val_loss={self.best_val_loss:.4f}, "
            f"total_time={self.total_training_time:.1f}s"
        )

        return self.history

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": self.best_val_loss,
                "config": getattr(self.model, "config", {}),
            }, path)

    def save_history(self, path: Path):
        """Save training history to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.history)
        df.to_csv(path, index=False)
        logger.info(f"Training history saved to {path}")

    def plot_training_curves(self, save_path: Optional[Path] = None):
        """Plot training and validation loss curves."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.history["epoch"], self.history["train_loss"],
                label="Train Loss", color="#6366F1", linewidth=2)
        ax.plot(self.history["epoch"], self.history["val_loss"],
                label="Val Loss", color="#EF4444", linewidth=2)

        # Mark best epoch
        best_epoch_idx = np.argmin(self.history["val_loss"])
        best_epoch = self.history["epoch"][best_epoch_idx]
        best_loss = self.history["val_loss"][best_epoch_idx]
        ax.scatter([best_epoch], [best_loss], color="#10B981", s=100, zorder=5,
                   label=f"Best (epoch {best_epoch}, loss={best_loss:.4f})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title(f"{self.model_name} — Training Curves", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return fig
