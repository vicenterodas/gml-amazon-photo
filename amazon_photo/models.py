"""MLP de atributos y GCN de compras conjuntas, sin entrenamiento.

Ambos clasificadores tienen dos capas, la misma dimensión oculta, dropout y
número de parámetros. La llamada es ``logits = model(x, edge_index)``. La MLP
ignora ``edge_index``; la GCN agrega vecinos con la normalización de Kipf y
Welling. La pérdida futura debe usar sólo las etiquetas de entrenamiento.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot, zeros


def _reset_linear(layer: nn.Linear) -> None:
    """Misma inicialización que ``GCNConv``: Glorot en pesos y sesgo en cero."""
    glorot(layer.weight)
    zeros(layer.bias)


class MLP(nn.Module):
    """Clasifica cada producto usando sólo su vector de características."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_linear(self.lin1)
        _reset_linear(self.lin2)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Se acepta edge_index para compartir la llamada con la GCN.
        del edge_index
        x = F.dropout(F.relu(self.lin1(x)), p=self.dropout, training=self.training)
        return self.lin2(x)


class GCN(nn.Module):
    """Agrega vecinos de la red de compras conjuntas antes de clasificar.

    ``cached`` queda desactivado: cada pasada normaliza el ``edge_index``
    recibido. Retirar conexiones no reutiliza una adyacencia anterior. Los
    autoenlaces de ``GCNConv`` conservan la característica propia, también en
    productos aislados.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False)
        self.dropout = dropout

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def build_classifier(kind: str, in_channels: int, hidden_channels: int,
                     out_channels: int, dropout: float = 0.5) -> nn.Module:
    """Construye ``mlp`` o ``gcn`` con la misma capacidad."""
    if kind == "mlp":
        return MLP(in_channels, hidden_channels, out_channels, dropout)
    if kind == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, dropout)
    raise ValueError(f"Clasificador desconocido: {kind}")
