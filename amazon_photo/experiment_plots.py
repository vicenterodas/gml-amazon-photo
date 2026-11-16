"""Visualizaciones de comparación, convergencia y robustez."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_experiment_figures(metrics, summary, graphs, confusion, class_names, seed, output):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"mlp": "#2563a6", "gcn": "#b64a29"}

    def save(fig, name):
        fig.savefig(figures / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    baseline = summary[(summary.drop_fraction == 0) & (summary.protocol != "inference")].set_index("model").loc[["mlp", "gcn"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        bars = ax.bar(["MLP", "GCN"], baseline[f"test_{metric}_mean"],
                      yerr=baseline[f"test_{metric}_std"].fillna(0), capsize=5,
                      color=[colors["mlp"], colors["gcn"]])
        ax.bar_label(bars, fmt="%.3f", padding=10)
        ax.set(ylim=(0, 1.05), ylabel=title, title=f"Prueba · {title}")
    fig.suptitle("Grafo completo · media ± desviación estándar entre semillas")
    save(fig, "baseline_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for column, kind in enumerate(("mlp", "gcn")):
        history = pd.read_csv(output / "histories" / f"{kind}_seed{seed}_drop0.csv")
        best_epoch = metrics[(metrics.model == kind) & (metrics.seed == seed) &
                             (metrics.drop_fraction == 0)].iloc[0].best_epoch
        for split, color in (("train", colors["mlp"]), ("val", colors["gcn"])):
            axes[0, column].plot(history.epoch, history[f"{split}_loss"], label=split, color=color)
            axes[1, column].plot(history.epoch, history[f"{split}_accuracy"], label=split, color=color)
        for ax in axes[:, column]:
            ax.axvline(best_epoch, linestyle=":", color="#666666", label="Checkpoint elegido")
            ax.set(xlabel="Época")
            ax.legend(fontsize=8)
        axes[0, column].set(title=kind.upper(), ylabel="Entropía cruzada")
        axes[1, column].set(ylabel="Accuracy", ylim=(0, 1.02))
    fig.suptitle(f"Convergencia con grafo completo · semilla prefijada {seed}")
    save(fig, "learning_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        lower_values, upper_values = [], []
        for protocol, label, color in (("inference", "GCN · pesos fijos", "#b64a29"),
                                        ("retrain", "GCN · reentrenada", "#479174")):
            frame = summary[(summary.model == "gcn") & (summary.protocol == protocol)].sort_values("drop_fraction")
            x = frame.drop_fraction.to_numpy() * 100
            mean = frame[f"test_{metric}_mean"].to_numpy()
            std = frame[f"test_{metric}_std"].fillna(0).to_numpy()
            lower_values.extend(mean - std)
            upper_values.extend(mean + std)
            ax.plot(x, mean, "o-", label=label, color=color)
            ax.fill_between(x, np.maximum(0, mean-std), np.minimum(1, mean+std), color=color, alpha=.15)
        mlp_mean = baseline.loc["mlp", f"test_{metric}_mean"]
        mlp_std = baseline.loc["mlp", f"test_{metric}_std"]
        mlp_std = 0 if pd.isna(mlp_std) else mlp_std
        lower_values.append(mlp_mean - mlp_std)
        upper_values.append(mlp_mean + mlp_std)
        ax.axhline(mlp_mean, linestyle="--", color=colors["mlp"], label="MLP · independiente del grafo")
        ax.axhspan(mlp_mean-mlp_std, mlp_mean+mlp_std, color=colors["mlp"], alpha=.08)
        ax.set(xlabel="Conexiones únicas eliminadas (%)", ylabel=title,
               ylim=(max(0, min(lower_values)-.03), min(1, max(upper_values)+.03)),
               xticks=[0, 25, 50, 75, 100])
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Robustez en prueba · media ± desviación estándar (no es IC)\nEjes verticales acotados para mostrar las diferencias")
    save(fig, "edge_robustness.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, metric, title in zip(axes, ("isolated_nodes", "components"), ("Nodos aislados", "Componentes conexas")):
        grouped = graphs.groupby("drop_fraction")[metric].agg(["mean", "min", "max"])
        x = grouped.index.to_numpy() * 100
        ax.plot(x, grouped["mean"], "o-", color="#2563a6")
        ax.fill_between(x, grouped["min"], grouped["max"], alpha=.15, color="#2563a6")
        ax.set(xlabel="Conexiones únicas eliminadas (%)", ylabel=title, title=title, yscale="log")
    fig.suptitle("Fragmentación del grafo · media y rango entre realizaciones\nEjes verticales logarítmicos")
    save(fig, "graph_fragmentation.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, kind in zip(axes, ("mlp", "gcn")):
        frame = confusion[confusion.model == kind]
        matrix = frame.pivot(index="true_class", columns="predicted_class", values="count").to_numpy()
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if normalized[i, j] > .5 else "black")
        ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
               xticklabels=np.arange(len(class_names)), yticklabels=class_names,
               xlabel="ID de clase predicha", title=kind.upper())
        fig.colorbar(im, ax=ax, fraction=.045, label="Proporción dentro de la clase real")
    fig.suptitle(f"Matriz de confusión en prueba · grafo completo · semilla prefijada {seed}\nTexto: conteos; color: proporción por fila")
    save(fig, "confusion_matrices.png")
