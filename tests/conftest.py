"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hkjc_nn.data import simulate_races


@pytest.fixture(scope="session")
def races() -> pd.DataFrame:
    """A small but realistic canonical frame."""
    return simulate_races(
        n_races=240,
        n_horses=45,
        seed=1234,
        start_date="2019-01-01",
        end_date="2023-12-31",
    )


@pytest.fixture(scope="session")
def race_ids(races: pd.DataFrame) -> np.ndarray:
    return races["race_id"].to_numpy()


@pytest.fixture(scope="session")
def synthetic_binary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A tiny, fully controlled dataset: three races, one obvious winner.

    The winner is always the runner with the highest ``x``.  A correct model
    must recover that ordering exactly.
    """
    groups = []
    X = []
    y = []
    for race in range(3):
        n = 4
        x = np.zeros((n, 2))
        x[:, 0] = np.array([3.0, 1.0, -1.0, -3.0])
        x[:, 1] = np.array([0.0, 0.5, -0.5, 0.0])
        target = np.zeros(n)
        target[0] = 1.0
        X.append(x)
        y.append(target)
        groups += [f"r{race}"] * n
    return np.vstack(X), np.concatenate(y), np.asarray(groups)
