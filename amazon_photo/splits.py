"""Máscaras estratificadas; las etiquetas sólo determinan la estratificación."""

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def validate_masks(masks: dict, labels) -> None:
    y = np.asarray(labels)
    if set(masks) != {"train_mask", "val_mask", "test_mask"}:
        raise ValueError("Se requieren exactamente train_mask, val_mask y test_mask")
    for name, mask in masks.items():
        if mask.dtype != torch.bool or tuple(mask.shape) != (len(y),):
            raise ValueError(f"Máscara inválida: {name}")
        if set(np.unique(y[mask.numpy()])) != set(np.unique(y)):
            raise ValueError(f"Faltan clases en {name}")
    membership = torch.stack(list(masks.values())).sum(dim=0)
    if not torch.all(membership == 1):
        raise ValueError("Las máscaras deben ser disjuntas y cubrir todos los nodos")
    for name, fraction in (("train_mask", .6), ("val_mask", .2), ("test_mask", .2)):
        if abs(int(masks[name].sum()) - len(y) * fraction) > 1:
            raise ValueError(f"Proporción incorrecta: {name}")
        # Redondeo estratificado en dos pasos: tolerancia de dos nodos por clase.
        actual = np.bincount(y[masks[name].numpy()], minlength=int(y.max()) + 1)
        expected = np.bincount(y) * fraction
        if np.any(np.abs(actual - expected) > 2):
            raise ValueError(f"Estratificación incorrecta: {name}")


def make_masks(labels, seed: int = 42) -> dict:
    y = np.asarray(labels)
    nodes = np.arange(len(y))
    train, remaining = train_test_split(
        nodes, train_size=.6, random_state=seed, stratify=y,
    )
    val, test = train_test_split(
        remaining, test_size=.5, random_state=seed, stratify=y[remaining],
    )
    masks = {}
    for name, indices in (("train_mask", train), ("val_mask", val), ("test_mask", test)):
        mask = torch.zeros(len(y), dtype=torch.bool)
        mask[torch.from_numpy(indices)] = True
        masks[name] = mask
    validate_masks(masks, y)
    return masks
