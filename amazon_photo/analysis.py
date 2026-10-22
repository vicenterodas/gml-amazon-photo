"""Estadísticas del grafo simple no dirigido y tablas de exploración."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def analyze_graph(data, masks: dict):
    n = data.num_nodes
    source, target = data.edge_index.numpy()
    y = data.y.numpy()
    # Con reciprocidad, sin duplicados ni lazos, source < target cuenta cada par una vez.
    unique = source < target
    u, v = source[unique], target[unique]
    degree = np.bincount(source, minlength=n)
    adjacency = csr_matrix((np.ones(len(source), dtype=np.int8), (source, target)), shape=(n, n))
    component_count, component_ids = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(component_ids)
    same_class = int(np.count_nonzero(y[u] == y[v]))
    summary = {
        "num_nodes": n,
        "num_stored_edges": data.num_edges,
        "num_undirected_edges": len(u),
        "num_features": data.num_node_features,
        "num_classes": int(np.unique(y).size),
        "density": float(2 * len(u) / (n * (n - 1))),
        "num_components": int(component_count),
        "largest_component_nodes": int(component_sizes.max()),
        "isolated_nodes": int(np.count_nonzero(degree == 0)),
        "same_class_undirected_edges": same_class,
        "edge_homophily": float(same_class / len(u)),
        "degree": {"min": int(degree.min()), "max": int(degree.max()),
                   "mean": float(degree.mean()), "median": float(np.median(degree)),
                   "std": float(degree.std()),
                   "p95": float(np.percentile(degree, 95))},
        "features": {"dtype": str(data.x.dtype), "min": float(data.x.min()),
                     "max": float(data.x.max()),
                     "zero_fraction": float((data.x == 0).sum()) / data.x.numel()},
        "split_sizes": {name: int(mask.sum()) for name, mask in masks.items()},
    }
    counts = np.bincount(y)
    classes = pd.DataFrame({"class_id": np.arange(len(counts)), "count": counts,
                            "fraction": counts / n})
    for name, mask in masks.items():
        classes[name.removesuffix("_mask")] = np.bincount(y[mask.numpy()], minlength=len(counts))
    degrees = pd.DataFrame({"degree": np.arange(len(np.bincount(degree))),
                            "count": np.bincount(degree)})
    components = pd.DataFrame({"component_id": np.arange(component_count), "num_nodes": component_sizes})
    return summary, classes, degrees, components
