"""Benter-style horse racing probability models, and the neural alternatives.

Public entry points:

``hkjc_nn.data.load_races``
    Load a canonical race frame (synthetic or from a real CSV extract).
``hkjc_nn.features.build_features``
    Turn a canonical frame into a causal feature table.
``hkjc_nn.models``
    Conditional logit, LightGBM, listwise MLP, field transformer, market blend.
``hkjc_nn.experiments.run_experiments``
    Run the whole A-F experiment matrix.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
