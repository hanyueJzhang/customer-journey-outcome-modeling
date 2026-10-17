"""Optional event-sequence embedding and clustering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score


def head_tail_sequence(events: Sequence[str], tokens_per_side: int = 200) -> str:
    """Keep the beginning and end of a long journey sequence."""
    tokens = [str(event) for event in events]
    if len(tokens) <= 2 * tokens_per_side:
        selected = tokens
    else:
        selected = tokens[:tokens_per_side] + tokens[-tokens_per_side:]
    return " ".join(selected)


def embed_sequences(
    sequences: Iterable[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Create normalized Sentence-BERT embeddings on demand."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install requirements-clustering.txt to create sequence embeddings."
        ) from exc
    model = SentenceTransformer(model_name)
    return np.asarray(
        model.encode(list(sequences), normalize_embeddings=True, show_progress_bar=False)
    )


@dataclass(frozen=True)
class ClusterCandidate:
    clusters: int
    silhouette: float
    davies_bouldin: float
    minimum_size: int
    maximum_size: int


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    reduced_embeddings: np.ndarray
    candidates: tuple[ClusterCandidate, ...]
    selected_clusters: int


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    candidate_clusters: Sequence[int] = (3, 4, 5),
    pca_components: int = 50,
    random_state: int = 42,
) -> ClusteringResult:
    """Evaluate K-means candidates after PCA and select by silhouette score."""
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix.")
    components = min(pca_components, matrix.shape[0] - 1, matrix.shape[1])
    if components < 1:
        raise ValueError("At least two embedding rows are required.")
    reduced = PCA(n_components=components, random_state=random_state).fit_transform(matrix)
    candidates: list[ClusterCandidate] = []
    candidate_labels: dict[int, np.ndarray] = {}
    for clusters in candidate_clusters:
        if clusters < 2 or clusters >= len(reduced):
            continue
        labels = KMeans(
            n_clusters=clusters,
            n_init=20,
            random_state=random_state,
        ).fit_predict(reduced)
        sizes = np.bincount(labels, minlength=clusters)
        candidates.append(
            ClusterCandidate(
                clusters=clusters,
                silhouette=float(silhouette_score(reduced, labels)),
                davies_bouldin=float(davies_bouldin_score(reduced, labels)),
                minimum_size=int(sizes.min()),
                maximum_size=int(sizes.max()),
            )
        )
        candidate_labels[clusters] = labels
    if not candidates:
        raise ValueError("No candidate cluster count is valid for this sample size.")
    selected = max(candidates, key=lambda candidate: candidate.silhouette).clusters
    return ClusteringResult(
        labels=candidate_labels[selected],
        reduced_embeddings=reduced,
        candidates=tuple(candidates),
        selected_clusters=selected,
    )

