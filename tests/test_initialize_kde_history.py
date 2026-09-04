import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import initialize_kde_history


class DummySampler:
    def __init__(self, j):
        self.j = j

    def fit(self, X):
        return self

    def sample(self, X):
        return np.zeros(len(X), dtype=float)


def test_initialize_kde_history_uses_contiguous_fold_blocks():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.arange(20, dtype=float)

    q, q_tilde = initialize_kde_history(
        X,
        y,
        [DummySampler(0)],
        [0],
        batches=[2],
        splits=5,
        batches_to_draw_randomly=20,
        resamplings=3,
        model=None,
        loss=lambda y_pred, y_true: float(np.mean((y_pred - y_true) ** 2)),
    )

    assert len(q[2]) == 10
    assert len(q_tilde[2][0]) == 10
    assert all(arr.shape == (3,) for arr in q_tilde[2][0])
