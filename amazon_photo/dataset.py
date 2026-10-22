"""Carga oficial e integridad del grafo, sin transformaciones adicionales."""

import hashlib
from pathlib import Path

import torch
from torch_geometric.datasets import Amazon
from torch_geometric.utils import is_undirected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_fingerprint(data) -> str:
    """Huella de x, edge_index e y, incluyendo forma y tipo."""
    digest = hashlib.sha256()
    for key in ("x", "edge_index", "y"):
        array = data[key].detach().cpu().contiguous().numpy()
        digest.update(f"{key}:{array.shape}:{array.dtype}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_photo(root: Path):
    """Descarga si falta; reutiliza la caché de PyG en ejecuciones posteriores."""
    return Amazon(root=str(root), name="Photo")


def validate_photo(data, num_classes: int) -> dict:
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise ValueError(f"Validación fallida: {name}")

    check("pyg_data_validate", data.validate(raise_on_error=True))
    check("x_shape", tuple(data.x.shape) == (7650, 745))
    check("edge_index_shape", tuple(data.edge_index.shape) == (2, 238162))
    check("y_shape", tuple(data.y.shape) == (7650,))
    check("x_float", data.x.is_floating_point())
    check("x_finite", torch.isfinite(data.x).all())
    check("y_integer", data.y.dtype == torch.long)
    check("edge_index_integer", data.edge_index.dtype == torch.long)
    check("edge_indices_in_range", (data.edge_index >= 0).all()
          and (data.edge_index < data.num_nodes).all())
    check("eight_classes", num_classes == 8)
    check("all_classes_present", torch.equal(torch.unique(data.y), torch.arange(8)))
    check("undirected", is_undirected(data.edge_index, num_nodes=data.num_nodes))
    check("no_self_loops", not data.has_self_loops())
    unique_edges = torch.unique(data.edge_index, dim=1).shape[1]
    check("no_duplicate_directed_entries", unique_edges == data.num_edges)
    return checks
