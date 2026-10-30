"""Deterministic embeddings.

Prithivi's embedding model is ``hash-embed-v1``: a feature-hashing encoder
implemented with :mod:`hashlib` and nothing else.

Why not a neural encoder? Two reasons, both about the target device. A neural
embedding model needs an ONNX runtime or a compiled extension — neither builds
on aarch64-Android without an NDK toolchain, and neither is worth its weight
on a phone. And a personal knowledge base is small and lexically oriented:
the question is almost always "what did that PDF say about X?", which is a
matching problem, not a semantics problem.

What ``hash-embed-v1`` actually is:

* the text is lowercased and split into word tokens, word bigrams, and
  character 3-grams;
* each token is hashed with BLAKE2b into one of ``dims`` buckets;
* a second hash bit decides the bucket's sign, which keeps unrelated tokens
  from all piling up positively;
* the vector is L2-normalised, so cosine similarity is a dot product.

The result is a stable, cheap, fully local lexical embedding: two texts about
the same topic share many tokens and therefore land close together. It is not
a semantic model — "how do I stop a leak" and "fix a dripping tap" will not
match — and the retrieval layer compensates with BM25-style full-text search
fused on top of the vector score (see
:mod:`prithivi.rag.retrieval`).

When a heavier encoder becomes available, the model name changes and old rows
stay valid, because every embedding row records the model that produced it.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Iterable, Sequence

__all__ = ["Embedder", "HashEmbedder", "hash_embed", "tokenize"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Stop words removed before hashing. They appear in almost every document and
#: so contribute almost nothing to similarity while dominating the buckets.
_STOP_WORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "in", "on", "at", "to", "for", "from", "by", "with", "without", "within", "into", "onto", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "done", "have", "has", "had", "having", "this", "that", "these", "those", "it", "its", "as", "so", "such", "than", "too", "very", "can", "will", "just", "should", "now", "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them", "his", "her", "their", "not", "no", "nor", "only", "own", "same", "s", "t", "don"]
)


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase word tokens, with stop words removed by default."""
    words = _TOKEN_RE.findall(text.lower())
    if keep_stopwords:
        return words
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _features(text: str) -> list[str]:
    """Tokens, bigrams and character trigrams of one document.

    Word features carry the topical signal; bigrams keep word order ("flood
    warning" versus "warning flood"); character trigrams make the encoder
    tolerant of typos and of morphological variants ("flooding", "flooded").
    """
    words = tokenize(text)
    features: list[str] = list(words)
    features.extend(f"{a}_{b}" for a, b in itertools.pairwise(words))
    for word in words:
        if len(word) >= 4:
            padded = f"^{word}$"
            features.extend(padded[i : i + 3] for i in range(len(padded) - 2))
    return features


def _bucket(feature: str, dims: int) -> tuple[int, float]:
    """Map a feature to ``(bucket, sign)`` using two independent hashes."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest[:4], "big") % dims
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def hash_embed(text: str, dims: int = 384) -> list[float]:
    """Embed one string into a unit-length vector of ``dims`` dimensions."""
    if dims <= 0:
        raise ValueError("dims must be positive")
    vector = [0.0] * dims
    features = _features(text)
    if not features:
        return vector
    for feature in features:
        index, sign = _bucket(feature, dims)
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return vector
    return [v / norm for v in vector]


class Embedder:
    """Interface every embedding backend implements."""

    #: Model identifier stored alongside each vector.
    model: str = "embedder"
    #: Dimensionality of the vectors this backend produces.
    dims: int = 384

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity between two texts."""
        a = self.embed(left)
        b = self.embed(right)
        return sum(x * y for x, y in zip(a, b, strict=True))


class HashEmbedder(Embedder):
    """The built-in, dependency-free encoder."""

    def __init__(self, dims: int = 384) -> None:
        if dims <= 0:
            raise ValueError("dims must be positive")
        self.dims = dims
        self.model = "hash-embed-v1"

    def embed(self, text: str) -> list[float]:
        return hash_embed(text, self.dims)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [hash_embed(text, self.dims) for text in texts]


def build_embedder(model: str = "hash-embed-v1", *, dims: int = 384) -> Embedder:
    """Build the embedder named by ``model``.

    Raises:
        ValueError: for an unknown model name, listing what is available. An
            unknown model must fail loudly rather than silently fall back to a
            different encoder — vectors from two encoders are not comparable.
    """
    if model in {"hash-embed-v1", "hash", "default"}:
        return HashEmbedder(dims)
    raise ValueError(f"unknown embedding model {model!r}; available: hash-embed-v1")


def mean_vector(vectors: Iterable[Sequence[float]]) -> list[float]:
    """Average of several vectors, re-normalised. Empty input gives zeros."""
    collected = [list(v) for v in vectors]
    if not collected:
        return []
    dims = len(collected[0])
    out = [0.0] * dims
    for vector in collected:
        if len(vector) != dims:
            raise ValueError("cannot average vectors of different lengths")
        for index, value in enumerate(vector):
            out[index] += value
    count = float(len(collected))
    out = [v / count for v in out]
    norm = math.sqrt(sum(v * v for v in out))
    if norm <= 0.0:
        return out
    return [v / norm for v in out]
