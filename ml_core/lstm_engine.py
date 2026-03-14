"""
ml_core/lstm_engine.py
======================
PyTorch LSTM-based sequence model for TradePulse.

Architecture overview
---------------------
Input  : (batch, seq_len, input_size)  — e.g. 30 days of OHLCV + technical
         indicators + sentiment
Hidden : one or more stacked (optionally bidirectional) LSTM layers with
         dropout, followed by a multi-head self-attention pooling layer that
         learns *which* time steps are most informative for the prediction.
Output : scalar probability in [0, 1]  — P(price direction = UP)

Improvements over the baseline
-------------------------------
* **Bidirectional LSTM** — each hidden state captures both past and future
  context within the look-back window, improving pattern recognition.
* **Multi-head self-attention pooling** — instead of discarding all but the
  last hidden state, attention weights every time step and produces a
  richer context vector.
* **Deeper MLP head** — a two-layer MLP with ReLU and dropout replaces the
  single linear layer for better non-linear decision boundaries.
* **ReduceLROnPlateau scheduler** — automatically halves the learning rate
  when validation loss plateaus, enabling finer convergence.
* **Weight decay (L2)** — added to the Adam optimiser for regularisation.
* **Label smoothing** — small smoothing on the BCE targets reduces
  overconfident predictions and improves calibration on live data.

The ``LSTMModel`` class exposes a standard ``fit`` / ``predict_proba``
interface that mirrors scikit-learn conventions so it integrates cleanly
with the meta-learner stacking layer.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Neural network definition
# ---------------------------------------------------------------------------


class _AttentionLSTMNet(nn.Module):
    """Bidirectional LSTM with multi-head self-attention pooling.

    Architecture
    ------------
    1. Input projection (optional) — linear layer that expands/compresses
       the input features to ``hidden_size`` before the LSTM.
    2. Stacked bidirectional LSTM layers with inter-layer dropout.
    3. Multi-head self-attention over the full sequence of LSTM outputs —
       every time step attends to every other, producing an
       attention-weighted context vector (mean-pooled).
    4. BatchNorm → Dropout → MLP head (two linear layers with ReLU) →
       Sigmoid.

    Parameters
    ----------
    input_size : int
        Number of features per time step.
    hidden_size : int
        LSTM units per direction per layer.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability used between LSTM layers and in the MLP head.
    bidirectional : bool
        If ``True``, uses a bidirectional LSTM.  The output dimension is
        ``hidden_size * 2``.
    n_attention_heads : int
        Number of attention heads.  Must evenly divide the LSTM output
        dimension (``hidden_size * num_directions``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
        n_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_size * num_directions

        # Ensure n_attention_heads divides lstm_out_dim; fall back to 1 if not
        if lstm_out_dim % n_attention_heads != 0:
            n_attention_heads = max(
                h for h in range(1, n_attention_heads + 1) if lstm_out_dim % h == 0
            )
            logger.warning(
                "n_attention_heads adjusted to %d to divide lstm_out_dim=%d evenly.",
                n_attention_heads,
                lstm_out_dim,
            )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Multi-head self-attention for sequence pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.batch_norm = nn.BatchNorm1d(lstm_out_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

        # Two-layer MLP classification head
        mlp_hidden = max(lstm_out_dim // 2, 16)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor  shape (batch,)  — probability in [0, 1]
        """
        # lstm_out: (batch, seq_len, lstm_out_dim)
        lstm_out, _ = self.lstm(x)

        # Self-attention: every time step attends to all others
        # attended: (batch, seq_len, lstm_out_dim)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Mean pool over the time dimension for a fixed-size representation
        pooled = attended.mean(dim=1)       # (batch, lstm_out_dim)

        normed = self.batch_norm(pooled)
        dropped = self.dropout_layer(normed)
        logit = self.fc(dropped)            # (batch, 1)
        return self.sigmoid(logit).squeeze(1)  # (batch,)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class LSTMModel:
    """Wrapper around ``_AttentionLSTMNet`` with a sklearn-like training interface.

    Parameters
    ----------
    input_size : int
        Number of features per time-step.  More features (OHLCV + technical
        indicators + sentiment) generally improve real-world accuracy.
    seq_len : int
        Number of look-back time steps (e.g. 30 for 30 trading days).
    hidden_size : int
        Number of LSTM units per direction per layer.  Default 128.
    num_layers : int
        Number of stacked LSTM layers.  Default 2.
    dropout : float
        Dropout probability applied inside LSTM layers and in the MLP head.
        Default 0.3.
    bidirectional : bool
        If ``True``, uses a bidirectional LSTM — each hidden state captures
        both backward and forward context within the sequence window.
        Default ``True``.
    n_attention_heads : int
        Number of parallel attention heads in the self-attention pooling
        layer.  Must evenly divide ``hidden_size * (2 if bidirectional else 1)``.
        Default 4.
    learning_rate : float
        Initial Adam optimiser learning rate.  Default 1e-3.
    weight_decay : float
        L2 regularisation coefficient for the Adam optimiser.  Default 1e-4.
    epochs : int
        Maximum training epochs.  Default 50.
    batch_size : int
        Mini-batch size.  Default 64.
    patience : int
        Early-stopping patience (epochs without validation-loss improvement
        before training halts).  Default 10.
    lr_scheduler_factor : float
        Factor by which the learning rate is reduced by ``ReduceLROnPlateau``
        when validation loss stagnates.  Default 0.5.
    lr_scheduler_patience : int
        Number of epochs with no improvement before the scheduler reduces
        the LR.  Default 5.
    label_smoothing : float
        Smoothing applied to binary targets: ``y = y * (1 - ε) + 0.5 * ε``.
        Reduces overconfident predictions.  Default 0.05.
    device : str or None
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  If ``None``, auto-detected.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int = 30,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        n_attention_heads: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 5,
        label_smoothing: float = 0.05,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.n_attention_heads = n_attention_heads
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.label_smoothing = label_smoothing
        self.random_state = random_state

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        torch.manual_seed(random_state)
        self.model_: Optional[_AttentionLSTMNet] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> _AttentionLSTMNet:
        return _AttentionLSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            n_attention_heads=self.n_attention_heads,
        ).to(self.device)

    @staticmethod
    def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(arr, dtype=dtype)

    @staticmethod
    def _smooth_labels(y: torch.Tensor, smoothing: float) -> torch.Tensor:
        """Apply label smoothing: y' = y * (1 - ε) + 0.5 * ε."""
        if smoothing <= 0.0:
            return y
        return y * (1.0 - smoothing) + 0.5 * smoothing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMModel":
        """Train the attention-LSTM on sequential data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, seq_len, input_size)
        y_train : np.ndarray, shape (n_samples,)  — binary labels {0, 1}
        X_val : np.ndarray or None
            Optional validation set for early stopping and LR scheduling.
        y_val : np.ndarray or None

        Returns
        -------
        self
        """
        self.model_ = self._build_model()

        # Adam with L2 weight decay for regularisation
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()

        # Build DataLoaders
        train_ds = TensorDataset(
            self._to_tensor(X_train),
            self._to_tensor(y_train),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        use_val = X_val is not None and y_val is not None
        val_loader = None
        if use_val:
            val_ds = TensorDataset(
                self._to_tensor(X_val),
                self._to_tensor(y_val),
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        # ReduceLROnPlateau halves the LR after lr_scheduler_patience
        # epochs with no validation-loss improvement (only if val set used)
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
            )
            if use_val
            else None
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[dict] = None

        for epoch in range(1, self.epochs + 1):
            # ---- Training pass ----
            self.model_.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # Apply label smoothing to targets
                y_smooth = self._smooth_labels(y_batch, self.label_smoothing)
                optimizer.zero_grad()
                preds = self.model_(X_batch)
                loss = criterion(preds, y_smooth)
                loss.backward()
                # Gradient clipping prevents exploding gradients in LSTMs
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(train_ds)

            # ---- Validation pass, LR scheduling & early stopping ----
            if use_val:
                val_loss = self._evaluate_loss(val_loader, criterion)
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %d/%d — train_loss: %.4f  val_loss: %.4f  lr: %.2e",
                    epoch, self.epochs, train_loss, val_loss, current_lr,
                )
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Deep-copy the best weights
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model_.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            "Early stopping triggered at epoch %d.", epoch
                        )
                        break
            else:
                logger.info(
                    "Epoch %d/%d — train_loss: %.4f",
                    epoch, self.epochs, train_loss,
                )

        # Restore best weights if validation was used
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Compute mean BCE loss over a DataLoader (no gradient tracking)."""
        self.model_.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.model_(X_batch)
                loss = criterion(preds, y_batch)
                total_loss += loss.item() * len(X_batch)
                total_samples += len(X_batch)
        return total_loss / total_samples if total_samples > 0 else 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the UP class for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, seq_len, input_size)

        Returns
        -------
        np.ndarray, shape (n_samples,)  — values in [0, 1]
        """
        if self.model_ is None:
            raise RuntimeError("LSTMModel has not been fitted yet.")
        self.model_.eval()
        tensor = self._to_tensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model_(tensor).cpu().numpy()
        return probs
