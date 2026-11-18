"""Visualizaciones de comparación, convergencia y robustez."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ("mlp", "gcn", "gat")
MODEL_LABELS = {"mlp": "MLP", "gcn": "GCN", "gat": "GAT"}
COLORS = {"mlp": "#2563a6", "gcn": "#b64a29", "gat": "#6d28d9"}
ROBUSTNESS_STYLES = {
    ("gcn", "inference"): ("#b64a29", "o-", "GCN · pesos fijos"),
    ("gcn", "retrain"): ("#479174", "o-", "GCN · reentrenada"),
    ("gat", "inference"): ("#6d28d9", "s-", "GAT · pesos fijos"),
    ("gat", "retrain"): ("#d97706", "s-", "GAT · reentrenada"),
}


def _present(frame):
    return [kind for kind in MODEL_ORDER if kind in set(frame["model"])]


def save_experiment_figures(metrics, summary, graphs, confusion, class_names, seed, output):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(figures / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    present = _present(summary)
    baseline = summary[(summary.drop_fraction == 0) & (summary.protocol != "inference")]
    baseline = baseline.drop_duplicates("model").set_index("model")
    baseline = baseline.loc[[kind for kind in present if kind in baseline.index]]
    labels = [MODEL_LABELS[kind] for kind in baseline.index]
    fig, axes = plt.subplots(1, 2, figsize=(4.2 + 1.4 * len(labels), 4.2), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        bars = ax.bar(labels, baseline[f"test_{metric}_mean"],
                      yerr=baseline[f"test_{metric}_std"].fillna(0), capsize=5,
                      color=[COLORS[kind] for kind in baseline.index])
        ax.bar_label(bars, fmt="%.3f", padding=10)
        ax.set(ylim=(0, 1.08), ylabel=title, title=f"Prueba · {title}")
    fig.suptitle("Grafo completo · media ± desviación estándar entre semillas")
    save(fig, "baseline_comparison.png")

    curve_models = [kind for kind in present
                    if (output / "histories" / f"{kind}_seed{seed}_drop0.csv").exists()]
    fig, axes = plt.subplots(2, len(curve_models), figsize=(4.2 * len(curve_models), 7),
                             layout="constrained", squeeze=False)
    for column, kind in enumerate(curve_models):
        history = pd.read_csv(output / "histories" / f"{kind}_seed{seed}_drop0.csv")
        chosen = metrics[(metrics.model == kind) & (metrics.seed == seed) &
                         (metrics.drop_fraction == 0) & (metrics.protocol != "inference")]
        best_epoch = chosen.iloc[0].best_epoch
        for split, color in (("train", COLORS["mlp"]), ("val", COLORS["gcn"])):
            axes[0, column].plot(history.epoch, history[f"{split}_loss"], label=split, color=color)
            axes[1, column].plot(history.epoch, history[f"{split}_accuracy"], label=split, color=color)
        for ax in axes[:, column]:
            ax.axvline(best_epoch, linestyle=":", color="#666666", label="Checkpoint elegido")
            ax.set(xlabel="Época")
            ax.legend(fontsize=8)
        axes[0, column].set(title=MODEL_LABELS[kind], ylabel="Entropía cruzada")
        axes[1, column].set(ylabel="Accuracy", ylim=(0, 1.02))
    fig.suptitle(f"Convergencia con grafo completo · semilla prefijada {seed}")
    save(fig, "learning_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")):
        lower_values, upper_values = [], []
        for kind in ("gcn", "gat"):
            for protocol in ("inference", "retrain"):
                frame = summary[(summary.model == kind) & (summary.protocol == protocol)]
                if frame.empty:
                    continue
                frame = frame.sort_values("drop_fraction")
                color, style, label = ROBUSTNESS_STYLES[(kind, protocol)]
                x = frame.drop_fraction.to_numpy() * 100
                mean = frame[f"test_{metric}_mean"].to_numpy()
                std = frame[f"test_{metric}_std"].fillna(0).to_numpy()
                lower_values.extend(mean - std)
                upper_values.extend(mean + std)
                ax.plot(x, mean, style, label=label, color=color)
                ax.fill_between(x, np.maximum(0, mean - std), np.minimum(1, mean + std),
                                color=color, alpha=.12)
        if "mlp" in baseline.index:
            mlp_mean = baseline.loc["mlp", f"test_{metric}_mean"]
            mlp_std = baseline.loc["mlp", f"test_{metric}_std"]
            mlp_std = 0 if pd.isna(mlp_std) else mlp_std
            lower_values.append(mlp_mean - mlp_std)
            upper_values.append(mlp_mean + mlp_std)
            ax.axhline(mlp_mean, linestyle="--", color=COLORS["mlp"],
                       label="MLP · independiente del grafo")
            ax.axhspan(mlp_mean - mlp_std, mlp_mean + mlp_std, color=COLORS["mlp"], alpha=.08)
        if lower_values:
            y_min = max(0, min(lower_values) - .03)
            y_max = min(1, max(upper_values) + .03)
            if y_max <= y_min:
                y_min, y_max = 0, 1
        else:
            y_min, y_max = 0, 1
        ax.set(xlabel="Conexiones únicas eliminadas (%)", ylabel=title,
               ylim=(y_min, y_max), xticks=[0, 25, 50, 75, 100])
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

    confusion_models = [kind for kind in present if kind in set(confusion["model"])]
    fig, axes = plt.subplots(1, len(confusion_models), figsize=(6 * len(confusion_models), 5.2),
                             layout="constrained", squeeze=False)
    for ax, kind in zip(axes[0], confusion_models):
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
               xlabel="ID de clase predicha", title=MODEL_LABELS[kind])
        fig.colorbar(im, ax=ax, fraction=.045, label="Proporción dentro de la clase real")
    fig.suptitle(f"Matriz de confusión en prueba · grafo completo · semilla prefijada {seed}\nTexto: conteos; color: proporción por fila")
    save(fig, "confusion_matrices.png")
