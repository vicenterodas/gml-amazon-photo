"""MLP, GCN y GAT para clasificar productos, sin entrenamiento.

Los tres clasificadores tienen dos capas, la misma dimensión oculta y el
mismo dropout. La llamada es ``logits = model(x, edge_index)``. La MLP ignora
``edge_index``. La GCN agrega vecinos con la normalización de Kipf y Welling.
La GAT agrega vecinos con atención multi-cabeza (Veličković et al.). MLP y
GCN tienen el mismo número de parámetros; la GAT añade sólo los vectores de
atención. La pérdida futura debe usar sólo las etiquetas de entrenamiento.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv
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


class GAT(nn.Module):
    """Agrega vecinos con atención, conservando el ancho oculto de la GCN.

    La primera capa usa ``heads`` cabezas de ``hidden_channels // heads``
    rasgos y las concatena, así la representación intermedia tiene el mismo
    tamaño que en la GCN. La segunda capa tiene una sola cabeza y produce los
    logits de clase, como en la configuración transductiva de Veličković et al.

    La no linealidad entre capas es ReLU y el dropout de características está
    en el mismo lugar que en la MLP y la GCN. El dropout de los coeficientes
    de atención usa esa misma probabilidad. La puntuación de cada vecino sigue
    siendo la LeakyReLU de pendiente 0,2 propia de ``GATConv``. No hay
    caché: cada pasada recalcula la atención sobre el ``edge_index`` recibido.
    Los autoenlaces conservan la característica propia, también en productos
    aislados. Con un único vecino —el autoenlace— el softmax vale 1 y la capa
    se reduce a su transformación lineal.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.5, heads: int = 8):
        super().__init__()
        if heads < 1:
            raise ValueError("heads debe ser positivo")
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels debe ser divisible por el número de cabezas")
        self.heads = heads
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads,
                             concat=True, dropout=dropout, add_self_loops=True)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False,
                             dropout=dropout, add_self_loops=True)
        self.dropout = dropout

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def build_classifier(kind: str, in_channels: int, hidden_channels: int,
                     out_channels: int, dropout: float = 0.5, heads: int = 8) -> nn.Module:
    """Construye ``mlp``, ``gcn`` o ``gat``.

    ``heads`` sólo afecta a la GAT. MLP y GCN comparten el número de
    parámetros lineales; la GAT suma los vectores de atención.
    """
    if kind == "mlp":
        return MLP(in_channels, hidden_channels, out_channels, dropout)
    if kind == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, dropout)
    if kind == "gat":
        return GAT(in_channels, hidden_channels, out_channels, dropout, heads)
    raise ValueError(f"Clasificador desconocido: {kind}")
