"""Entrenamiento full-batch; recibe sólo las etiquetas de train y validación."""

from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    patience: int = 50
    learning_rate: float = 0.01
    weight_decay: float = 5e-4

    def __post_init__(self):
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs y patience deben ser positivos")
        if (not np.isfinite([self.learning_rate, self.weight_decay]).all()
                or self.learning_rate <= 0 or self.weight_decay < 0):
            raise ValueError("learning_rate debe ser positivo y weight_decay no negativo")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_epoch(model, optimizer, x, edge_index, train_mask, train_targets):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x, edge_index)
    loss = F.cross_entropy(logits[train_mask], train_targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Pérdida de entrenamiento no finita")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def fit(model, x, edge_index, train_mask, train_targets, val_mask, val_targets,
        config: TrainingConfig):
    """Minimiza CE en train; restaura el checkpoint de menor CE de validación.

    Esta función no recibe etiquetas de prueba. Los empates conservan el primer
    checkpoint. La parada cuenta épocas consecutivas sin mejora estricta.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, config.epochs + 1):
        optimization_loss = train_epoch(model, optimizer, x, edge_index, train_mask, train_targets)
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            train_loss = float(F.cross_entropy(logits[train_mask], train_targets))
            val_loss = float(F.cross_entropy(logits[val_mask], val_targets))
            train_accuracy = float((logits[train_mask].argmax(1) == train_targets).float().mean())
            val_accuracy = float((logits[val_mask].argmax(1) == val_targets).float().mean())
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError("Pérdida de evaluación no finita")
        history.append({"epoch": epoch, "optimization_loss": optimization_loss,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history, {"best_epoch": best_epoch, "epochs_run": len(history),
                     "best_val_loss": best_loss}


def classification_metrics(logits, targets, num_classes: int) -> dict:
    """Evaluación sin gradientes; macro-F1 da el mismo peso a cada categoría."""
    predictions = logits.argmax(1).detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    return {"loss": float(F.cross_entropy(logits.detach(), targets)),
            "accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, labels=list(range(num_classes)),
                                       average="macro", zero_division=0))}
