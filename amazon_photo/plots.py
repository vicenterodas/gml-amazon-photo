"""Figuras estáticas; backend sin interfaz para ejecuciones reproducibles."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figures(classes, degrees, components, summary, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.bar(classes.class_id, classes["count"], color="#2563a6")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set(xlabel="ID de categoría", ylabel="Número de productos", title="Amazon Photo · distribución de categorías",
           xticks=classes.class_id, ylim=(0, classes["count"].max() * 1.16))
    save(fig, "class_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    degree_values = np.repeat(degrees.degree.to_numpy(), degrees["count"].to_numpy())
    axes[0].hist(degree_values, bins=50, color="#2563a6")
    axes[0].set(xlabel="Grado (vecinos únicos)", ylabel="Número de nodos", title="Distribución de grados")
    # La cola acumulada permite ver grados raros sin ocultar nodos de grado cero.
    tail = np.cumsum(degrees["count"].to_numpy()[::-1])[::-1] / summary["num_nodes"]
    axes[1].step(degrees.degree, tail, where="post", color="#b64a29")
    axes[1].set(yscale="log", xlabel="Grado k", ylabel="Proporción de nodos con grado ≥ k", title="Cola acumulada · eje vertical logarítmico")
    save(fig, "degree_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    sizes, counts = np.unique(components.num_nodes, return_counts=True)
    axes[0].bar(np.arange(len(sizes)), counts, color="#2563a6")
    axes[0].set(xticks=np.arange(len(sizes)), xticklabels=sizes, xlabel="Tamaño de componente (nodos)",
                ylabel="Número de componentes", title=f"{summary['num_components']} componentes conexas", yscale="log")
    h = summary["edge_homophily"]
    bars = axes[1].bar(["Misma clase", "Clases distintas"], [h, 1-h], color=["#2563a6", "#b64a29"])
    axes[1].bar_label(bars, labels=[f"{h:.2%}", f"{1-h:.2%}"], padding=5)
    axes[1].set(ylabel="Proporción de aristas únicas", ylim=(0, 1), title="Homofilia por arista")
    save(fig, "components_homophily.png")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bottom = np.zeros(len(classes))
    for name, label, color in (("train", "Entrenamiento", "#2563a6"), ("val", "Validación", "#d68b2c"), ("test", "Prueba", "#479174")):
        values = classes[name].to_numpy() / classes["count"].to_numpy()
        ax.bar(classes.class_id, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set(xticks=classes.class_id, xlabel="ID de categoría", ylabel="Proporción dentro de cada clase", ylim=(0, 1),
           title="Particiones estratificadas · 60% / 20% / 20%")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=3, frameon=False)
    save(fig, "split_distribution.png")
