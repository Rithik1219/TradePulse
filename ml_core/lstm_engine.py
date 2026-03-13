"""
ml_core/lstm_engine.py
======================
PyTorch LSTM-based sequence model for TradePulse.

Architecture overview
---------------------
Input  : (batch, seq_len, input_size)  — e.g. 30 days of OHLCV + sentiment
Hidden : one or more stacked LSTM layers with optional dropout
Output : scalar probability in [0, 1]  — P(price direction = UP)

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


class _LSTMNet(nn.Module):
    """Internal PyTorch ``Module`` — consumed by ``LSTMModel``."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Stack multiple LSTM layers; apply dropout between layers
        # (PyTorch only applies dropout between layers, not after the last one)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Batch normalisation on the final hidden state for training stability
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        # Additional dropout before the classification head
        self.dropout = nn.Dropout(p=dropout)
        # Binary classification head → single logit
        self.fc = nn.Linear(hidden_size, 1)
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
        # lstm_out: (batch, seq_len, hidden_size)
        # We use only the *last* time-step's hidden state
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]          # (batch, hidden_size)
        normed = self.batch_norm(last_hidden)
        dropped = self.dropout(normed)
        logit = self.fc(dropped)                  # (batch, 1)
        prob = self.sigmoid(logit).squeeze(1)     # (batch,)
        return prob


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class LSTMModel:
    """Wrapper around ``_LSTMNet`` with a sklearn-like training interface.

    Parameters
    ----------
    input_size : int
        Number of features per time-step (e.g. 6 for OHLCV + sentiment).
    seq_len : int
        Number of look-back time steps (e.g. 30 for 30 days).
    hidden_size : int
        Number of LSTM units per layer.  Default 128.
    num_layers : int
        Number of stacked LSTM layers.  Default 2.
    dropout : float
        Dropout probability applied inside and after the LSTM.  Default 0.3.
    learning_rate : float
        Adam optimiser learning rate.  Default 1e-3.
    epochs : int
        Maximum training epochs.  Default 50.
    batch_size : int
        Mini-batch size.  Default 64.
    patience : int
        Early-stopping patience (epochs without validation-loss improvement
        before training halts).  Default 10.
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
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
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
        self.model_: Optional[_LSTMNet] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> _LSTMNet:
        return _LSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    @staticmethod
    def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(arr, dtype=dtype)

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
        """Train the LSTM on sequential data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, seq_len, input_size)
        y_train : np.ndarray, shape (n_samples,)  — binary labels {0, 1}
        X_val : np.ndarray or None
            Optional validation set for early stopping.
        y_val : np.ndarray or None

        Returns
        -------
        self
        """
        self.model_ = self._build_model()
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate
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
        if use_val:
            val_ds = TensorDataset(
                self._to_tensor(X_val),
                self._to_tensor(y_val),
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

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
                optimizer.zero_grad()
                preds = self.model_(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                # Gradient clipping prevents exploding gradients in LSTMs
                nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(train_ds)

            # ---- Validation pass & early stopping ----
            if use_val:
                val_loss = self._evaluate_loss(val_loader, criterion)
                logger.info(
                    "Epoch %d/%d — train_loss: %.4f  val_loss: %.4f",
                    epoch, self.epochs, train_loss, val_loss,
                )
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
