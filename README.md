# Amazon Photo · IIC3641

Proyecto del curso **IIC3641 – Aprendizaje Automático basado en Grafos**, Pontificia Universidad Católica de Chile.

**Objetivo:** comparar una MLP y una GCN para clasificar productos de Amazon Photo, y luego evaluar cómo cambia el desempeño de la GCN al eliminar progresivamente conexiones.

**Estado actual:** preparación y validación del dataset completas. El proyecto descarga los datos, comprueba su integridad, genera estadísticas y figuras, y crea máscaras estratificadas 60/20/20. **Todavía no se han implementado ni entrenado modelos ni eliminado aristas.**

## Inicio rápido

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m amazon_photo.prepare --seed 42
```

En la primera ejecución se descarga Amazon Photo desde GitHub. Las siguientes reutilizan la caché. No hace falta GPU ni instalar extensiones opcionales de PyG para esta etapa.

El comando genera:

| Ruta | Contenido |
|---|---|
| `data/Photo/` | Datos originales y caché procesada de PyG |
| `data/splits/photo_seed42.npz` | Máscaras booleanas de entrenamiento, validación y prueba |
| `results/dataset_report.json` | Dimensiones, estadísticas, verificaciones, versiones y huellas SHA-256 |
| `results/tables/` | Distribuciones de clases, grados y componentes en CSV |
| `results/figures/` | Cuatro figuras PNG |

`data/` y `.venv/` están excluidos de Git. Los resultados del análisis sí se incluyen. Al volver a ejecutar con las mismas rutas se regeneran las máscaras y los resultados, sin modificar los archivos originales del dataset.

Para comprobar el código y las dependencias:

```bash
python -m unittest discover -s tests -v
python -m pip check
```

Entorno verificado: **Python 3.14.7, macOS ARM64, CPU**. `requirements.txt` y `pyproject.toml` fijan las dependencias directas. Para reproducir también las versiones transitivas del entorno utilizado, instalar con `python -m pip install -r requirements-lock.txt` en lugar de `requirements.txt`. La instalación desde cero y otras versiones de Python o plataformas no se han verificado.

### Opciones de ejecución

```bash
python -m amazon_photo.prepare --data-root data --results-dir results --seed 42
```

`--data-root` controla la caché y la carpeta de máscaras; `--results-dir` controla el informe, las tablas y las figuras. La semilla predeterminada es `42`. Si falla la descarga, la carga o una verificación, el programa termina con un error.

Si Matplotlib no puede escribir su caché, ejecutar:

```bash
MPLCONFIGDIR=/tmp/gml-amazon-matplotlib python -m amazon_photo.prepare --seed 42
```

## Origen y representación

Se utiliza exactamente `torch_geometric.datasets.Amazon(root="data", name="Photo")`, sin `transform` ni `pre_transform`. Según la [documentación oficial de PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Amazon.html), los nodos son productos, las conexiones representan compras conjuntas y las características provienen de reseñas representadas como bolsa de palabras. Las etiquetas corresponden a categorías de producto.

El dataset se distribuye con [gnn-benchmark](https://github.com/shchur/gnn-benchmark) y se describe en [Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/abs/1811.05868). El informe JSON registra la URL exacta utilizada y las huellas SHA-256.

La [carga oficial de PyG](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/amazon.html) lee el NPZ y convierte las conexiones a no dirigidas. En la versión instalada, su lector también binariza las características positivas y quita lazos. Son operaciones propias de la representación estándar de PyG; nuestro código no aplica transformaciones adicionales ni escribe sobre los archivos originales. Se verifica que la caché procesada coincida exactamente con una nueva lectura oficial del NPZ.

`edge_index` almacena cada conexión no dirigida dos veces: `(u, v)` y `(v, u)`. Por eso **238.162 entradas almacenadas equivalen a 119.081 conexiones únicas**. Las estadísticas de aristas únicas usan `u < v`, después de comprobar reciprocidad, ausencia de duplicados y ausencia de lazos. El grado es la cantidad de vecinos únicos; no se divide por dos el grado calculado sobre los orígenes de `edge_index`.

## Resultados verificados

Ejecución del **21 de octubre de 2026**, semilla **42**. Resultados medidos en los datos reales, disponibles íntegramente en [`results/dataset_report.json`](results/dataset_report.json).

| Medida | Resultado |
|---|---:|
| `x` | `(7650, 745)`, `float32`, valores finitos entre 0 y 1 |
| `edge_index` | `(2, 238162)`, `int64`, índices válidos |
| `y` | `(7650,)`, `int64`, clases 0–7 presentes |
| Grafo | No dirigido, sin lazos ni entradas duplicadas |
| Nodos / conexiones únicas | 7.650 / 119.081 |
| Densidad | 0,0040701121 ≈ 0,4070% |
| Componentes conexas | 136 |
| Componente mayor | 7.487 nodos |
| Nodos aislados | 115 |
| Grado mínimo / mediano / medio / máximo | 0 / 22 / 31,1323 / 1.434 |
| Percentil 95 del grado | 87 |
| Aristas únicas con la misma clase | 98.509 |
| Homofilia por arista | 0,82724364 ≈ 82,7244% |
| Características iguales a cero | 65,2602% |

Para un grafo simple no dirigido, la densidad se calcula como `2m / (n(n−1))`, donde `m` cuenta pares únicos. La homofilia es `sum(y[u] == y[v]) / m`, también sobre pares únicos; no es una medida ajustada por el desequilibrio de clases. Las componentes incluyen los nodos aislados. No se filtra la componente mayor.

Los nombres de categorías se obtienen de `class_names` en el NPZ original, respetando su orden:

| ID | Categoría | Total | Entrenamiento | Validación | Prueba |
|---|---|---:|---:|---:|---:|
| 0 | Film Photography | 369 | 221 | 74 | 74 |
| 1 | Digital Cameras | 1.686 | 1.012 | 337 | 337 |
| 2 | Binoculars & Scopes | 703 | 422 | 140 | 141 |
| 3 | Lenses | 915 | 549 | 183 | 183 |
| 4 | Tripods & Monopods | 882 | 529 | 177 | 176 |
| 5 | Video Surveillance | 823 | 494 | 164 | 165 |
| 6 | Lighting & Studio | 1.941 | 1.164 | 389 | 388 |
| 7 | Flashes | 331 | 199 | 66 | 66 |
| **Total** | | **7.650** | **4.590** | **1.530** | **1.530** |

La tabla completa generada por el programa está en [`results/tables/class_distribution.csv`](results/tables/class_distribution.csv).

Figuras guardadas en `results/figures/`:

- [`class_distribution.png`](results/figures/class_distribution.png): categorías y cantidades.
- [`degree_distribution.png`](results/figures/degree_distribution.png): histograma y cola acumulada de grados, incluyendo grado cero.
- [`components_homophily.png`](results/figures/components_homophily.png): tamaños de componentes y homofilia.
- [`split_distribution.png`](results/figures/split_distribution.png): proporciones por clase en las tres particiones.

## Particiones y protocolo transductivo

El problema es **clasificación transductiva de nodos**: todos los nodos, sus características y el grafo completo están disponibles durante el futuro entrenamiento de la GCN, pero sólo las etiquetas de entrenamiento pueden participar en la pérdida y en las actualizaciones de parámetros. Para la MLP se utilizarán las mismas características y máscaras. No se generan características a partir de etiquetas.

Se usa `train_test_split` de Scikit-learn dos veces con `stratify` y `random_state=42`: primero 60% entrenamiento y 40% restante, luego mitad validación y mitad prueba sobre ese restante. Las máscaras booleanas tienen forma `(7650,)`, son disjuntas, cubren todos los nodos y mantienen las ocho clases. El redondeo por clase puede variar en uno o dos nodos. Se comprueban sus proporciones, cobertura y reproducibilidad, incluyendo la lectura posterior al guardado.

Las máscaras se guardan en **`data/splits/photo_seed42.npz`**, con claves `train_mask`, `val_mask`, `test_mask`. No se incorporan a la caché original. Ejemplo para una etapa posterior:

```python
import numpy as np
import torch
from torch_geometric.datasets import Amazon

data = Amazon(root="data", name="Photo")[0]
with np.load("data/splits/photo_seed42.npz", allow_pickle=False) as saved:
    for key in ("train_mask", "val_mask", "test_mask"):
        data[key] = torch.from_numpy(saved[key].copy())
# En el entrenamiento futuro: pérdida sólo en logits[data.train_mask]
# contra data.y[data.train_mask]. No hay entrenamiento en esta etapa.
```

Las etiquetas de validación sirven exclusivamente para evaluación y selección de hiperparámetros o parada temprana, sin gradientes. Las etiquetas de prueba se reservan para la evaluación final y no pueden orientar entrenamiento ni selección. La estratificación y las estadísticas descriptivas solicitadas —como la homofilia global— sí consultan etiquetas, pero no crean entradas del modelo ni definen sus parámetros. No se debe usar este análisis de etiquetas de prueba para elegir una arquitectura o ajustar hiperparámetros.

En los futuros experimentos, mantener fijas las máscaras para comparar MLP/GCN y niveles de eliminación. La eliminación de una conexión no dirigida deberá retirar ambos sentidos de `edge_index`; no se ha implementado todavía.

## Estructura

```text
amazon_photo/
  dataset.py         # Carga, huellas e integridad
  splits.py          # Creación y validación de máscaras
  analysis.py        # Estadísticas y tablas
  plots.py           # Cuatro figuras con Matplotlib
  prepare.py         # CLI que coordina la preparación
tests/
  test_preparation.py
data/                # Ignorado por Git; caché PyG y máscaras regenerables
results/
  dataset_report.json
  figures/           # Figuras PNG del análisis real
  tables/            # CSV de categorías, grados y componentes
inicio.py            # Script inicial del repositorio, conservado
pyproject.toml
requirements.txt
requirements-lock.txt
.gitignore
README.md
```

No se incluyen datasets, entornos virtuales, secretos ni temporales en Git. Se conservan informes, tablas y figuras para revisar esta etapa. El código está separado por responsabilidades para incorporar después modelos y experimentos sin modificar la carga original.

## Verificaciones realizadas

- **20 comprobaciones del pipeline superadas**: estructura, tipos, finitud, etiquetas, índices, reciprocidad, lazos, duplicados, coincidencia raw/caché, máscaras reproducibles y correctamente guardadas, y conservación de tensores y archivos originales mediante huellas.
- **2 pruebas automatizadas superadas**: particiones estratificadas y rechazo de solapamiento; cálculo de pares únicos, densidad, componentes y homofilia en un grafo pequeño con un nodo aislado.
- `pip check`: **sin dependencias incompatibles**.
- Una **descarga nueva** desde GitHub y su procesamiento con PyG produjeron los mismos tensores que la copia local. Las huellas están registradas en [`results/dataset_report.json`](results/dataset_report.json).
- Dos ejecuciones con semilla 42 produjeron las mismas huellas de los tensores y del contenido de las máscaras. Las cuatro figuras se revisaron visualmente.

## Limitaciones

- Se reutilizaron las dependencias ya instaladas; no se comprobó una instalación desde cero ni la compatibilidad con otras plataformas.
- PyTorch emite un `FutureWarning` sobre `torch.jit.script` con Python 3.14. Esta etapa no utiliza JIT y terminó correctamente; su compatibilidad deberá revisarse antes de usar compilación.
- El NPZ original permanece intacto. La representación cargada corresponde al procesamiento estándar de PyG explicado arriba.
- La URL de origen apunta a `master`; las huellas permiten detectar cambios futuros. El hash binario de `data.pt` puede variar entre versiones de serialización aunque sus tensores coincidan.
- No hay métricas predictivas: el entrenamiento de MLP/GCN y los experimentos de eliminación de aristas corresponden a las siguientes etapas.
