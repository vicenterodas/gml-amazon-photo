import torch
from torch_geometric.datasets import Amazon
from torch_geometric.utils import is_undirected

dataset = Amazon(root="data", name="Photo")
data = dataset[0]

print("Productos:", data.num_nodes)
print("Conexiones:", data.num_edges)
print("Características:", data.num_node_features)
print("Categorías:", dataset.num_classes)
print("Distribución de clases:", torch.bincount(data.y))
print("Grafo no dirigido:", is_undirected(data.edge_index))
print("Máscaras disponibles:", [
    key for key in ["train_mask", "val_mask", "test_mask"]
    if key in data
])

assert data.x.shape == (7650, 745)
assert data.edge_index.shape == (2, 238162)
assert data.y.shape == (7650,)
assert dataset.num_classes == 8
assert torch.isfinite(data.x).all()
assert data.y.min() >= 0
assert data.y.max() < 8

print("Validación básica superada")