"""
trainer.py
Training loop with early stopping, AMP mixed precision, and checkpoint saving.
"""

import os
import time
import json
import logging
import torch
from torch.cuda.amp import GradScaler, autocast
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.stop = False

    def step(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.stop


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler,
                 device, cfg: dict, start_epoch: int = 0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.start_epoch = start_epoch

        self.ckpt_dir = cfg["paths"]["checkpoint_dir"]
        self.use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
        self.grad_clip = cfg["training"]["gradient_clip"]
        self.epochs = cfg["training"]["epochs"]

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)

        self.scaler = GradScaler(enabled=self.use_amp)
        self.early_stop = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    # ── One Epoch ─────────────────────────────────────────────────────────
    def _train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(X)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()

            # Gradient clipping prevents exploding gradients in LSTM
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self, loader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                with autocast(enabled=self.use_amp):
                    logits = self.model(X)
                    loss = self.criterion(logits, y)

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += len(y)
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ── Full Training Loop ─────────────────────────────────────────────────
    def fit(self, train_loader, val_loader):
        best_val_loss = float("inf")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_acc = self._val_epoch(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch+1:03d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | lr={lr:.6f} | {elapsed:.1f}s"
            )
            print(
                f"Epoch {epoch+1:03d}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  lr={lr:.2e}  [{elapsed:.1f}s]"
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, "best.pt")
                logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

            # Save latest checkpoint (for resume)
            self._save_checkpoint(epoch, val_loss, "latest.pt")

            # Early stopping
            if self.early_stop.step(val_loss):
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                break

        # Save training history
        history_path = os.path.join(self.cfg["paths"]["log_dir"], "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[INFO] Training history saved to {history_path}")

    def _save_checkpoint(self, epoch: int, val_loss: float, filename: str):
        path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
        }, path)
