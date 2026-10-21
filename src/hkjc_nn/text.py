"""Text branch: free-text commentary -> a small dense vector.

The report's architecture note is important and easy to get wrong: **the text
should be encoded by a text encoder, not by the tabular model.**  A tabular
transformer consumes numbers and categories; it has nothing to say about
Chinese race commentary.

This module provides the seam.  The default encoder is a TF-IDF + truncated
SVD pipeline, which is cheap, deterministic and dependency-free, and already
captures a lot of the signal in templated commentary (it can tell you that a
note is positive or negative).  The output is compressed to 8-16 dimensions,
exactly as the report recommends, because racing datasets are small and high
dimensional text features overfit fast.

To use a real encoder instead (Qwen, BERT, any sentence-transformer), implement
:class:`TextEncoder` and pass it in — nothing else in the codebase changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["TextEncoder", "TfidfSvdTextEncoder", "build_text_matrix"]


class TextEncoder:
    """Interface for anything that turns a set of documents into vectors."""

    def fit(self, documents: list[str]) -> "TextEncoder":  # pragma: no cover - interface
        raise NotImplementedError

    def transform(self, documents: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    @property
    def dim(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class TfidfSvdTextEncoder(TextEncoder):
    """TF-IDF over character n-grams, compressed with truncated SVD.

    Character n-grams (rather than word tokens) are used because the corpus
    is Chinese race commentary: word segmentation needs a dictionary, character
    bigrams work without one.
    """

    n_components: int = 12
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 3)
    random_state: int = 0

    def __post_init__(self) -> None:
        self._vectorizer = None
        self._svd = None
        self._fitted = False

    def fit(self, documents: list[str]) -> "TfidfSvdTextEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        clean = [d if isinstance(d, str) and d.strip() else "" for d in documents]
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=3,
            sublinear_tf=True,
        )
        # The SVD cannot have more components than the matrix has columns, and
        # a sparse corpus (most horses have no veterinary note at all) can end
        # up with a very small vocabulary.  Shrink the request rather than
        # crash.
        try:
            counts = self._vectorizer.fit_transform(clean)
            n_features = counts.shape[1]
        except ValueError:
            n_features = 0
        if n_features == 0:
            self.n_components = 0
            self._svd = None
            self._fitted = True
            return self
        if self.n_components > n_features:
            self.n_components = max(1, n_features)

        pipe = make_pipeline(
            self._vectorizer,
            TruncatedSVD(n_components=self.n_components, random_state=self.random_state),
            Normalizer(copy=False),
        )
        self._svd = pipe.fit(clean)
        self._fitted = True
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("encoder is not fitted")
        if self.n_components == 0 or self._svd is None:
            # An empty vocabulary is not an error: it just means the corpus
            # carries no text.  Return an empty matrix rather than crashing.
            return np.zeros((len(documents), 0), dtype=float)
        clean = [d if isinstance(d, str) and d.strip() else "" for d in documents]
        return np.asarray(self._svd.transform(clean), dtype=float)

    @property
    def dim(self) -> int:
        return self.n_components


def build_text_matrix(
    frame: pd.DataFrame,
    text_columns: tuple[str, ...] = ("notes_text", "vet_report_text"),
    *,
    encoder: TextEncoder | None = None,
    fit_frame: pd.DataFrame | None = None,
    n_components: int = 12,
    prefix: str = "txt",
) -> tuple[np.ndarray, list[str]]:
    """Encode the text columns of ``frame`` into a dense matrix.

    Parameters
    ----------
    frame
        Frame to encode.
    text_columns
        Which text columns to use.  Missing columns are ignored.
    encoder
        Custom encoder.  Defaults to :class:`TfidfSvdTextEncoder`.
    fit_frame
        Frame the encoder is fitted on.  Use the *training* split so the
        vocabulary is not influenced by validation or test data.
    """
    cols = [c for c in text_columns if c in frame.columns]
    if not cols:
        return np.zeros((len(frame), 0)), []

    def combine(df: pd.DataFrame) -> list[str]:
        parts = []
        for col in cols:
            values = df[col].fillna("").astype(str) if col in df.columns else pd.Series([""] * len(df))
            parts.append(values)
        joined = pd.concat(parts, axis=1).agg(" ".join, axis=1)
        return [d if d.strip() else "" for d in joined.tolist()]

    fit_frame = fit_frame if fit_frame is not None else frame
    encoder = encoder or TfidfSvdTextEncoder(n_components=n_components)
    encoder.fit(combine(fit_frame))
    matrix = encoder.transform(combine(frame))

    names = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
    return matrix, names