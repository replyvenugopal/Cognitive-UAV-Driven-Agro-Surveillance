"""Cognitive field segmentation / adaptive spatial zoning (Section 3.4, Eq. 17).

Clusters observations into stress-homogeneous zones based on the similarity
of their stress-evolution pattern, rather than a fixed geometric grid.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def cluster_into_zones(
    stress_features: np.ndarray,
    n_zones: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Partition rows into `n_zones` stress-homogeneous clusters (Eq. 17).

    Z = Cluster(X_z^t, S_z^t)

    Parameters
    ----------
    stress_features: array of shape (N, D) combining UAV-derived stress
        indicators and soil variables for each observation.
    n_zones: number of adaptive zones to form (Z in the paper's notation).

    Returns
    -------
    zone_ids: array of shape (N,) with integer zone assignments in
    [0, n_zones).
    """
    stress_features = np.asarray(stress_features, dtype=float)
    n_zones = max(1, min(n_zones, len(stress_features)))
    km = KMeans(n_clusters=n_zones, random_state=random_state, n_init=10)
    return km.fit_predict(stress_features)


def zone_adjacency(zone_ids: np.ndarray, k_neighbors: int = 3) -> dict[int, list[int]]:
    """Approximate a neighbor graph N(z) over zones for the diffusion module.

    Because the tabular dataset has no explicit spatial coordinates, we
    approximate zone adjacency using proximity in feature-centroid space
    (zones whose mean stress-feature profile is most similar are treated as
    "neighboring" for the purposes of the diffusion equations, Eq. 26-27).
    This is documented as a modeling assumption in docs/dataset.md; users
    with true GPS-referenced UAV flights should replace this with a real
    spatial adjacency (e.g. Delaunay triangulation of zone centroids).
    """
    unique_zones = sorted(set(zone_ids.tolist()))
    n = len(unique_zones)
    k_neighbors = max(1, min(k_neighbors, n - 1)) if n > 1 else 0
    # Ring topology fallback keeps this deterministic and dependency-free.
    adjacency: dict[int, list[int]] = {}
    for i, z in enumerate(unique_zones):
        neighbors = []
        for offset in range(1, k_neighbors + 1):
            neighbors.append(unique_zones[(i + offset) % n])
            neighbors.append(unique_zones[(i - offset) % n])
        adjacency[z] = sorted(set(neighbors) - {z})
    return adjacency
