"""Training callbacks for MANTRA.

Callbacks allow monitoring and controlling the training process.
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class Callback:
    """Base class for training callbacks."""

    def on_epoch_end(
        self,
        epoch: int,
        loss: float,
        history: list[float],
    ) -> bool:
        """Called at the end of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number
        loss : float
            Loss for this epoch
        history : List[float]
            Full loss history

        Returns
        -------
        bool
            Whether to stop training early
        """
        return False

    def on_train_end(self, history: list[float]) -> None:
        """Called when training ends.

        Parameters
        ----------
        history : List[float]
            Full loss history
        """


class EarlyStoppingCallback(Callback):
    """Stop training when loss stops improving.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait for improvement, by default 50
    min_delta : float, optional
        Minimum change to qualify as improvement, by default 1e-4
    """

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(
        self,
        epoch: int,
        loss: float,
        history: list[float],
    ) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info("Early stopping: no improvement for %d epochs", self.patience)
            return True

        return False


class CheckpointCallback(Callback):
    """Save model checkpoints during training.

    Parameters
    ----------
    path : str
        Directory to save checkpoints
    every_n_epochs : int, optional
        Save every N epochs, by default 100
    """

    def __init__(
        self,
        path: str,
        every_n_epochs: int = 100,
    ) -> None:
        self.path = Path(path)
        self.every_n_epochs = every_n_epochs
        self.path.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        epoch: int,
        loss: float,
        history: list[float],
    ) -> bool:
        if epoch > 0 and epoch % self.every_n_epochs == 0:
            checkpoint_path = self.path / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "history": history,
                },
                checkpoint_path,
            )
            logger.info("Saved checkpoint to %s", checkpoint_path)

        return False


class LogCallback(Callback):
    """Log training progress.

    Parameters
    ----------
    log_every : int, optional
        Log every N epochs, by default 100
    """

    def __init__(self, log_every: int = 100) -> None:
        self.log_every = log_every

    def on_epoch_end(
        self,
        epoch: int,
        loss: float,
        history: list[float],
    ) -> bool:
        if epoch > 0 and epoch % self.log_every == 0:
            logger.info("Epoch %d: ELBO = %.4f", epoch, loss)

        return False

    def on_train_end(self, history: list[float]) -> None:
        logger.info("Training complete. Final ELBO: %.4f", history[-1])
