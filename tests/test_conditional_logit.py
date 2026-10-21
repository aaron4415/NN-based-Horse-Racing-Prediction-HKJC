"""Tests for the conditional logit (model A).

The gradient check is the important one.  A sign error in the analytic
gradient does not raise — it silently makes the optimiser walk *up* the loss
and abort with ``ABNORMAL_TERMINATION_IN_LNSRCH``, leaving an untrained
model that predicts a uniform distribution.  It is exactly the kind of bug
that survives a casual smoke test, so it is pinned down here.
"""

from __future__ import annotations

import numpy as np
import pytest

from hkjc_nn.models.base import group_normalise
from hkjc_nn.models.conditional_logit import ConditionalLogit


def test_analytic_gradient_matches_finite_differences(synthetic_binary):
    X, y, groups = synthetic_binary
    model = ConditionalLogit(l2=0.5)
    Xs = model._prepare_fit(X)
    slices = model._race_slices(groups)
    winners = [int(np.flatnonzero(y[idx] > 0.5)[0]) for idx in slices]

    rng = np.random.default_rng(0)
    beta = rng.normal(0.0, 0.4, X.shape[1])
    _, grad = model._objective(beta, Xs, slices, winners)

    h = 1e-6
    for j in range(X.shape[1]):
        step = np.zeros_like(beta)
        step[j] = h
        f_plus, _ = model._objective(beta + step, Xs, slices, winners)
        f_minus, _ = model._objective(beta - step, Xs, slices, winners)
        numeric = (f_plus - f_minus) / (2 * h)
        assert abs(numeric - grad[j]) < 1e-4 * max(1.0, abs(numeric))


def test_model_converges(synthetic_binary):
    X, y, groups = synthetic_binary
    model = ConditionalLogit(l2=0.1).fit(X, y, groups)
    assert model.converged_, "L-BFGS-B failed to converge"
    assert model.n_iter_ > 0
    assert model.coef_ is not None


def test_probabilities_sum_to_one_within_each_race(synthetic_binary):
    X, y, groups = synthetic_binary
    model = ConditionalLogit(l2=0.1).fit(X, y, groups)
    p = model.predict_proba(X, groups)
    for race in np.unique(groups):
        mask = groups == race
        assert np.isclose(p[mask].sum(), 1.0)


def test_recovers_the_signal_direction(synthetic_binary):
    """The strongest runner must get the highest probability."""
    X, y, groups = synthetic_binary
    model = ConditionalLogit(l2=0.01).fit(X, y, groups)
    p = model.predict_proba(X, groups)
    for race in np.unique(groups):
        mask = groups == race
        assert int(np.argmax(p[mask])) == int(np.argmax(y[mask]))


def test_l2_shrinks_coefficients(synthetic_binary):
    X, y, groups = synthetic_binary
    loose = ConditionalLogit(l2=1e-6).fit(X, y, groups)
    tight = ConditionalLogit(l2=100.0).fit(X, y, groups)
    assert np.abs(tight.coef_).sum() < np.abs(loose.coef_).sum()


def test_group_normalise_handles_zero_rows():
    values = np.array([0.0, 0.0, 1.0, 3.0])
    groups = np.array(["a", "a", "b", "b"])
    out = group_normalise(values, groups)
    assert np.isclose(out[0], 0.5)
    assert np.isclose(out[1], 0.5)
    assert np.isclose(out[2], 0.25)
    assert np.isclose(out[3], 0.75)


def test_predict_before_fit_raises(synthetic_binary):
    X, y, groups = synthetic_binary
    with pytest.raises(RuntimeError):
        ConditionalLogit().predict_proba(X, groups)
