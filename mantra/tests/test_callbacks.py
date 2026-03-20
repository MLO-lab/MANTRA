"""Tests for MANTRA training callbacks."""

from mantra.inference.callbacks import Callback, EarlyStoppingCallback, LogCallback


class TestBaseCallback:
    """Tests for the Callback base class."""

    def test_callable(self):
        """Test that Callback instances are callable with history list."""
        cb = Callback()
        result = cb([100.0, 90.0, 80.0])
        assert result is False

    def test_on_epoch_end_receives_correct_args(self):
        """Test that __call__ correctly extracts epoch and loss."""
        received = {}

        class RecordingCallback(Callback):
            def on_epoch_end(self, epoch, loss, history):
                received["epoch"] = epoch
                received["loss"] = loss
                received["history"] = history
                return False

        cb = RecordingCallback()
        history = [100.0, 90.0, 80.0]
        cb(history)

        assert received["epoch"] == 2
        assert received["loss"] == 80.0
        assert received["history"] is history


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_no_stop_when_improving(self):
        """Test that training continues when loss is improving."""
        cb = EarlyStoppingCallback(patience=3)
        history = []
        for loss in [100.0, 90.0, 80.0, 70.0]:
            history.append(loss)
            assert cb(history) is False

    def test_stop_after_patience(self):
        """Test that early stopping triggers after patience epochs."""
        cb = EarlyStoppingCallback(patience=3, min_delta=0.0)
        history = []
        # First improve, then plateau
        for loss in [100.0, 90.0, 80.0]:
            history.append(loss)
            assert cb(history) is False

        # Now stagnate for patience epochs
        for _ in range(2):
            history.append(80.0)
            assert cb(history) is False

        history.append(80.0)
        assert cb(history) is True

    def test_counter_resets_on_improvement(self):
        """Test that counter resets when loss improves."""
        cb = EarlyStoppingCallback(patience=3)
        history = []

        # Stagnate for 2 epochs
        for loss in [100.0, 100.0, 100.0]:
            history.append(loss)
            cb(history)
        assert cb.counter == 2

        # Improve - counter should reset
        history.append(50.0)
        cb(history)
        assert cb.counter == 0


class TestLogCallback:
    """Tests for LogCallback."""

    def test_no_error(self):
        """Test that LogCallback runs without error."""
        cb = LogCallback(log_every=2)
        history = []
        for loss in [100.0, 90.0, 80.0, 70.0, 60.0]:
            history.append(loss)
            result = cb(history)
            assert result is False


class TestCallbacksInFit:
    """Tests for callbacks used in model.fit()."""

    def test_early_stopping_in_fit(self, mantra_model, set_seeds):
        """Test that EarlyStoppingCallback works in fit()."""
        cb = EarlyStoppingCallback(patience=3, min_delta=1e10)
        history, stopped = mantra_model.fit(
            n_epochs=100,
            n_particles=1,
            learning_rate=0.01,
            optimizer="adam",
            callbacks=[cb],
            verbose=False,
            seed=42,
        )
        # With min_delta=1e10, it should stop very early
        assert stopped is True
        assert len(history) < 100

    def test_log_callback_in_fit(self, mantra_model, set_seeds):
        """Test that LogCallback works in fit()."""
        cb = LogCallback(log_every=2)
        history, _ = mantra_model.fit(
            n_epochs=5,
            n_particles=1,
            learning_rate=0.01,
            optimizer="adam",
            callbacks=[cb],
            verbose=False,
            seed=42,
        )
        assert len(history) == 5
