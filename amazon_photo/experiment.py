"""Comparación MLP/GCN/GAT y robustez: python -m amazon_photo.experiment."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from .dataset import load_photo, sha256_file, tensor_fingerprint, validate_photo
from .models import build_classifier
from .perturbation import edge_subsets, graph_statistics
from .splits import make_masks, validate_masks
from .training import TrainingConfig, classification_metrics, fit, seed_everything
from .experiment_plots import save_experiment_figures


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/experiments"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--drop-fractions", type=float, nargs="+", default=[0, .1, .25, .5, .75, .9, 1])
    parser.add_argument("--models", nargs="+", default=["mlp", "gcn", "gat"],
                        choices=("mlp", "gcn", "gat"))
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    args.drop_fractions = sorted(set([0.] + args.drop_fractions))
    if any(not np.isfinite(f) or not 0 <= f <= 1 for f in args.drop_fractions):
        parser.error("Las fracciones eliminadas deben estar entre 0 y 1")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 10000 for s in args.seeds):
        parser.error("Las semillas deben ser únicas, no negativas y menores que 2**32 - 10000")
    if args.hidden_channels < 1 or not 0 <= args.dropout < 1 or args.threads < 1:
        parser.error("hidden-channels y threads deben ser positivos; dropout debe estar en [0, 1)")
    if len(set(args.models)) != len(args.models):
        parser.error("Los modelos no deben repetirse")
    args.models = [kind for kind in ("mlp", "gcn", "gat") if kind in args.models]
    if "gat" in args.models and (args.heads < 1 or args.hidden_channels % args.heads != 0):
        parser.error("Con GAT, heads debe ser positivo y dividir a hidden-channels")
    return args


def main():
    args = parse_args()
    config = TrainingConfig(args.epochs, args.patience, args.learning_rate, args.weight_decay)
    # CPU con un número de hilos explícito: no se cambia de dispositivo implícitamente.
    torch.set_num_threads(args.threads)
    dataset = load_photo(args.data_root)
    data = dataset[0]
    validate_photo(data, dataset.num_classes)
    original = tensor_fingerprint(data)
    protected_files = [Path(dataset.raw_paths[0]), Path(dataset.processed_paths[0]),
                       args.data_root / "splits" / f"photo_seed{args.split_seed}.npz"]
    if not protected_files[-1].exists():
        raise FileNotFoundError(f"Ejecuta primero python -m amazon_photo.prepare --seed {args.split_seed}")
    hashes_before = {str(path): sha256_file(path) for path in protected_files}
    with np.load(protected_files[-1], allow_pickle=False) as saved:
        masks = {key: torch.from_numpy(saved[key].copy()) for key in ("train_mask", "val_mask", "test_mask")}
    validate_masks(masks, data.y.numpy())
    expected = make_masks(data.y.numpy(), args.split_seed)
    if not all(torch.equal(masks[k], expected[k]) for k in masks):
        raise ValueError("Las máscaras guardadas no corresponden a la semilla solicitada")
    masks_hash = hashlib.sha256(b"".join(masks[k].numpy().tobytes() for k in sorted(masks))).hexdigest()
    with np.load(dataset.raw_paths[0], allow_pickle=False) as raw:
        class_names = raw["class_names"].astype(str).tolist()

    for directory in (args.results_dir, args.artifacts_dir, args.results_dir / "histories"):
        directory.mkdir(parents=True, exist_ok=True)
    rows, graph_rows, confusion_rows = [], [], []
    parameter_counts = {}
    start = time.perf_counter()
    for seed in args.seeds:
        edge_seed = seed + 10000
        graphs = edge_subsets(data.edge_index, data.num_nodes, args.drop_fractions, edge_seed)
        for fraction, edges in graphs.items():
            stats = graph_statistics(edges, data.num_nodes)
            graph_rows.append({"seed": seed, "edge_seed": edge_seed, "drop_fraction": fraction,
                               "actual_drop_fraction": 1 - stats["stored_edges"] / data.num_edges, **stats})

        def train(kind, fraction):
            seed_everything(seed)
            model = build_classifier(kind, data.num_features, args.hidden_channels,
                                     dataset.num_classes, args.dropout, args.heads)
            parameter_counts[kind] = sum(p.numel() for p in model.parameters())
            fit_start = time.perf_counter()
            history, metadata = fit(model, data.x, graphs[fraction], masks["train_mask"],
                                    data.y[masks["train_mask"]], masks["val_mask"],
                                    data.y[masks["val_mask"]], config)
            metadata["fit_seconds"] = time.perf_counter() - fit_start
            metadata["state_sha256"] = state_fingerprint(model)
            tag = f"{kind}_seed{seed}_drop{fraction:g}"
            pd.DataFrame(history).to_csv(args.results_dir / "histories" / f"{tag}.csv", index=False)
            torch.save({"model_state_dict": model.state_dict(), "model": kind,
                        "in_channels": data.num_features, "hidden_channels": args.hidden_channels,
                        "out_channels": dataset.num_classes, "dropout": args.dropout,
                        "heads": args.heads if kind == "gat" else None,
                        "training_config": asdict(config), "seed": seed, "edge_seed": edge_seed,
                        "drop_fraction": fraction, "split_seed": args.split_seed,
                        "tensors_sha256": original, "masks_content_sha256": masks_hash,
                        **metadata}, args.artifacts_dir / f"{tag}.pt")
            print(f"{tag}: mejor época {metadata['best_epoch']}/{metadata['epochs_run']}, "
                  f"val_loss={metadata['best_val_loss']:.4f}, {metadata['fit_seconds']:.1f}s", flush=True)
            return model, metadata

        # Todas las decisiones de entrenamiento de esta semilla ya están cerradas.
        # El test se consulta ahora, exclusivamente para las métricas finales.
        conditions = []
        if "mlp" in args.models:
            mlp, mlp_metadata = train("mlp", 0.)
            conditions.append(("mlp", "attributes", 0., mlp, mlp_metadata))
        for kind in ("gcn", "gat"):
            if kind not in args.models:
                continue
            full_model, full_metadata = train(kind, 0.)
            trained = {0.: (full_model, full_metadata)}
            for fraction in args.drop_fractions:
                if fraction > 0:
                    trained[fraction] = train(kind, fraction)
            for fraction, (model, metadata) in trained.items():
                conditions.append((kind, "retrain", fraction, model, metadata))
                conditions.append((kind, "inference", fraction, full_model, full_metadata))
        for kind, protocol, fraction, model, metadata in conditions:
            with torch.no_grad():
                logits = model(data.x, graphs[fraction])
            row = {"model": kind, "protocol": protocol, "seed": seed, "edge_seed": edge_seed,
                   "drop_fraction": fraction, "parameters": parameter_counts[kind], **metadata}
            for split, mask in masks.items():
                prefix = split.removesuffix("_mask")
                scores = classification_metrics(logits[mask], data.y[mask], dataset.num_classes)
                row.update({f"{prefix}_{metric}": value for metric, value in scores.items()})
            rows.append(row)
            # Semilla representativa fijada antes de mirar test; no se escoge la mejor.
            if seed == args.seeds[0] and fraction == 0 and protocol != "inference":
                matrix = confusion_matrix(data.y[masks["test_mask"]].numpy(),
                                          logits[masks["test_mask"]].argmax(1).numpy(),
                                          labels=np.arange(dataset.num_classes))
                for true_class in range(dataset.num_classes):
                    for predicted in range(dataset.num_classes):
                        confusion_rows.append({"model": kind, "seed": seed, "true_class": true_class,
                                               "predicted_class": predicted,
                                               "count": int(matrix[true_class, predicted])})
        # Progreso recuperable si se interrumpe una ejecución larga.
        pd.DataFrame(rows).to_csv(args.results_dir / "metrics.csv", index=False)
        pd.DataFrame(graph_rows).to_csv(args.results_dir / "graphs.csv", index=False)

    metrics = pd.DataFrame(rows)
    scores = [f"{split}_{metric}" for split in ("train", "val", "test")
              for metric in ("loss", "accuracy", "macro_f1")]
    grouped = metrics.groupby(["model", "protocol", "drop_fraction"])[scores].agg(["mean", "std", "count"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary = grouped.reset_index()
    summary.to_csv(args.results_dir / "summary.csv", index=False)
    paired_columns = ["model", "seed", "protocol", "drop_fraction",
                      "delta_accuracy_pp", "delta_macro_f1_pp"]
    paired_rows = []
    if (metrics.model == "mlp").any():
        mlp_scores = metrics[metrics.model == "mlp"][["seed", "test_accuracy", "test_macro_f1"]]
        for kind in ("gcn", "gat"):
            subset = metrics[metrics.model == kind]
            if subset.empty:
                continue
            paired = subset.merge(mlp_scores, on="seed", suffixes=(f"_{kind}", "_mlp"))
            for metric in ("accuracy", "macro_f1"):
                paired[f"delta_{metric}_pp"] = 100 * (
                    paired[f"test_{metric}_{kind}"] - paired["test_" + metric + "_mlp"])
            paired_rows.append(paired[paired_columns])
    paired_frame = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame(columns=paired_columns)
    paired_frame.to_csv(args.results_dir / "paired_differences.csv", index=False)
    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(args.results_dir / "confusion_matrices.csv", index=False)
    checks = {"original_tensors_unchanged": tensor_fingerprint(data) == original,
              "original_files_unchanged": all(sha256_file(path) == hashes_before[str(path)] for path in protected_files),
              "masks_match_split_seed": True,
              "equal_parameter_counts": (
                  parameter_counts["mlp"] == parameter_counts["gcn"]
                  if {"mlp", "gcn"} <= parameter_counts.keys() else True),
              "gat_attention_parameters": (
                  parameter_counts["gat"] - parameter_counts["gcn"]
                  == 2 * args.hidden_channels + 2 * dataset.num_classes
                  if {"gat", "gcn"} <= parameter_counts.keys() else True)}
    if not all(checks.values()):
        raise RuntimeError(f"Falló la comprobación final: {checks}")
    manifest = {"training_config": asdict(config), "models": args.models,
                "hidden_channels": args.hidden_channels, "gat_heads": args.heads,
                "dropout": args.dropout, "seeds": args.seeds, "split_seed": args.split_seed,
                "drop_fractions": args.drop_fractions, "edge_seed_rule": "training_seed + 10000",
                "selection": "minimum validation cross-entropy; first checkpoint wins ties",
                "protocols": {"attributes": "MLP entrenada una vez por semilla; ignora el grafo",
                              "inference": "GCN o GAT entrenada con grafo completo; pesos fijos ante eliminación",
                              "retrain": "GCN o GAT reinicializada y entrenada en cada grafo incompleto"},
                "gat_design": ("Primera capa: cabezas concatenadas hasta hidden_channels. "
                               "Segunda capa: una cabeza hacia las clases. ReLU y dropout de "
                               "características iguales a MLP y GCN; dropout de atención con la "
                               "misma probabilidad. Los parámetros extra son los vectores de atención."),
                "edge_removal": "uniform undirected pairs without replacement; nested subsets; floor(fraction * m)",
                "features": "PyG original, sin transformaciones adicionales",
                "device": "cpu", "threads": args.threads, "deterministic_algorithms": True,
                "tensors_sha256": original, "masks_content_sha256": masks_hash,
                "protected_files_sha256": hashes_before, "checks": checks,
                "parameter_counts": parameter_counts, "class_names": class_names,
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "packages": {p: importlib.metadata.version(p) for p in
                                             ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "matplotlib", "scipy")}},
                "elapsed_seconds": time.perf_counter() - start}
    (args.results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_experiment_figures(metrics, summary, pd.DataFrame(graph_rows), confusion, class_names,
                            args.seeds[0], args.results_dir)
    print(f"Finalizado: {args.results_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
