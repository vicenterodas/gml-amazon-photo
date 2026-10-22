"""Ejecutar: python -m amazon_photo.prepare --seed 42."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch_geometric.io import read_npz

from .analysis import analyze_graph
from .dataset import load_photo, sha256_file, tensor_fingerprint, validate_photo
from .plots import save_figures
from .splits import make_masks, validate_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = load_photo(args.data_root)
    data = dataset[0]
    raw_path = Path(dataset.raw_paths[0])
    processed_path = Path(dataset.processed_paths[0])
    original_hashes = {"raw_sha256": sha256_file(raw_path),
                       "processed_sha256": sha256_file(processed_path)}
    fingerprint = tensor_fingerprint(data)
    checks = validate_photo(data, dataset.num_classes)

    # Detecta cachés procesadas distintas de la lectura oficial del archivo raw.
    raw_data = read_npz(str(raw_path), to_undirected=True)
    checks["cache_matches_raw_official_loader"] = all(
        torch.equal(data[key], raw_data[key]) for key in ("x", "edge_index", "y")
    )
    if not checks["cache_matches_raw_official_loader"]:
        raise ValueError("La caché no coincide con el raw cargado mediante PyG")
    del raw_data

    masks = make_masks(data.y.numpy(), seed=args.seed)
    repeated = make_masks(data.y.numpy(), seed=args.seed)
    checks["split_reproducible"] = all(torch.equal(masks[k], repeated[k]) for k in masks)
    if not checks["split_reproducible"]:
        raise ValueError("Partición no reproducible")
    summary, classes, degrees, components = analyze_graph(data, masks)
    with np.load(raw_path, allow_pickle=False) as raw:
        class_names = raw["class_names"].astype(str)
    if len(class_names) != dataset.num_classes:
        raise ValueError("El número de nombres de clases no coincide con las etiquetas")
    classes.insert(1, "class_name", class_names)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    classes.to_csv(tables_dir / "class_distribution.csv", index=False)
    degrees.to_csv(tables_dir / "degree_distribution.csv", index=False)
    components.to_csv(tables_dir / "components.csv", index=False)
    save_figures(classes, degrees, components, summary, args.results_dir / "figures")

    split_dir = args.data_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    mask_path = split_dir / f"photo_seed{args.seed}.npz"
    np.savez_compressed(mask_path, **{k: v.numpy() for k, v in masks.items()})
    with np.load(mask_path, allow_pickle=False) as saved:
        restored = {k: torch.from_numpy(saved[k].copy()) for k in masks}
    validate_masks(restored, data.y.numpy())
    checks["masks_saved_correctly"] = all(torch.equal(masks[k], restored[k]) for k in masks)
    checks["original_tensors_unchanged"] = fingerprint == tensor_fingerprint(data)
    checks["raw_file_unchanged"] = original_hashes["raw_sha256"] == sha256_file(raw_path)
    checks["processed_file_unchanged"] = original_hashes["processed_sha256"] == sha256_file(processed_path)
    if not all(checks.values()):
        raise ValueError("Falló la verificación final de integridad")

    report = {
        "dataset": "torch_geometric.datasets.Amazon(name='Photo')",
        "seed": args.seed,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {p: importlib.metadata.version(p) for p in
                                     ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "matplotlib", "scipy")}},
        "provenance": {"raw_url": dataset.url + raw_path.name, **original_hashes,
                       "tensors_sha256": fingerprint,
                       "masks_content_sha256": hashlib.sha256(b"".join(masks[k].numpy().tobytes() for k in sorted(masks))).hexdigest()},
        "shapes": {k: list(data[k].shape) for k in ("x", "edge_index", "y")},
        "checks": checks,
        "statistics": summary,
    }
    (args.results_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps({"checks_passed": len(checks), **summary}, indent=2, ensure_ascii=False))
    print(f"Informe: {args.results_dir / 'dataset_report.json'}")
    print(f"Máscaras: {mask_path}")


if __name__ == "__main__":
    main()
