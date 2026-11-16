"""Eliminación uniforme de pares no dirigidos, anidada y sin usar etiquetas."""

import hashlib

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import is_undirected


def edge_subsets(edge_index, num_nodes: int, fractions, seed: int):
    if not is_undirected(edge_index, num_nodes=num_nodes):
        raise ValueError("Se requiere un grafo no dirigido")
    source, target = edge_index
    if (source == target).any():
        raise ValueError("El grafo de entrada debe estar libre de lazos")
    pairs = edge_index[:, source < target]
    if torch.unique(pairs, dim=1).shape[1] != pairs.shape[1]:
        raise ValueError("El grafo de entrada contiene conexiones duplicadas")
    order = np.random.default_rng(seed).permutation(pairs.shape[1])
    subsets = {}
    for fraction in fractions:
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("La fracción eliminada debe estar entre 0 y 1")
        removed = int(np.floor(fraction * pairs.shape[1]))
        keep = torch.ones(pairs.shape[1], dtype=torch.bool)
        keep[torch.from_numpy(order[:removed])] = False
        retained = pairs[:, keep]
        edges = torch.cat((retained, retained.flip(0)), dim=1)
        edges = edges[:, torch.argsort(edges[0] * num_nodes + edges[1])]
        # Conserva exactamente el orden original cuando no se elimina nada.
        subsets[fraction] = edge_index.clone() if removed == 0 else edges
    return subsets


def graph_statistics(edge_index, num_nodes: int) -> dict:
    source, target = edge_index.numpy()
    degree = np.bincount(source, minlength=num_nodes)
    adjacency = csr_matrix((np.ones(len(source), dtype=np.int8), (source, target)),
                           shape=(num_nodes, num_nodes))
    count, ids = connected_components(adjacency, directed=False)
    return {"stored_edges": len(source), "unique_edges": len(source) // 2,
            "isolated_nodes": int((degree == 0).sum()), "components": int(count),
            "largest_component": int(np.bincount(ids).max()),
            "edges_sha256": hashlib.sha256(edge_index.contiguous().numpy().tobytes()).hexdigest()}
