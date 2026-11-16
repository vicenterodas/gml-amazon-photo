# Amazon Photo · IIC3641

Proyecto del curso **IIC3641 – Aprendizaje Automático basado en Grafos**, Pontificia Universidad Católica de Chile.

**Objetivo:** comparar una MLP y una GCN para clasificar productos de Amazon Photo, y luego evaluar cómo cambia el desempeño de la GCN al eliminar progresivamente conexiones.

El proyecto está organizado en tres etapas de implementación:

1. **Datos:** descarga, validación, exploración y máscaras estratificadas 60/20/20.
2. **Modelos:** MLP y GCN de dos capas, con igual dimensión oculta, dropout y número de parámetros.
3. **Experimentos:** entrenamiento, comparación con el baseline MLP y evaluación de robustez al eliminar conexiones, tanto con pesos fijos como con reentrenamiento.

Estas etapas de código son distintas de los hitos del curso (propuesta, avance y entrega final). La entrega final requiere una evaluación con métricas frente a un baseline. Los experimentos y su interpretación están en [`results/experiments/REPORT.md`](results/experiments/REPORT.md).

## Inicio rápido

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m amazon_photo.prepare --seed 42
python -m amazon_photo.experiment
```

En la primera ejecución se descarga Amazon Photo desde GitHub. Las siguientes reutilizan la caché. No hace falta GPU ni instalar extensiones opcionales de PyG para esta etapa.

Estos comandos generan:

| Ruta | Contenido |
|---|---|
| `data/Photo/` | Datos originales y caché procesada de PyG |
| `data/splits/photo_seed42.npz` | Máscaras booleanas de entrenamiento, validación y prueba |
| `results/dataset_report.json` | Dimensiones, estadísticas, verificaciones, versiones y huellas SHA-256 |
| `results/tables/` | Distribuciones de clases, grados y componentes en CSV |
| `results/figures/` | Cuatro figuras PNG |
| `results/experiments/` | Métricas por semilla, resúmenes, curvas, matrices de confusión y análisis de robustez |
| `artifacts/experiments/` | Checkpoints del mejor modelo según validación, por condición y semilla |

`data/`, `artifacts/` y `.venv/` están excluidos de Git. Los informes, tablas y figuras sí se incluyen. Al volver a ejecutar con las mismas rutas se regeneran los resultados y checkpoints de esa configuración, sin modificar los archivos originales del dataset. Usar carpetas nuevas al cambiar la configuración para conservar experimentos anteriores.

Para comprobar el código y las dependencias:

```bash
python -m unittest discover -s tests -v
python -m pip check
```

Entorno verificado: **Python 3.14.7, macOS ARM64, CPU**. `requirements.txt` y `pyproject.toml` fijan las dependencias directas. Para reproducir también las versiones transitivas del entorno utilizado, instalar con `python -m pip install -r requirements-lock.txt` en lugar de `requirements.txt`. La instalación desde cero y otras versiones de Python o plataformas no se han verificado.

### Opciones de preparación de datos

```bash
python -m amazon_photo.prepare --data-root data --results-dir results --seed 42
```

`--data-root` controla la caché y la carpeta de máscaras; `--results-dir` controla el informe, las tablas y las figuras. La semilla predeterminada es `42`. Si falla la descarga, la carga o una verificación, el programa termina con un error.

Si Matplotlib no puede escribir su caché, ejecutar:

```bash
MPLCONFIGDIR=/tmp/gml-amazon-matplotlib python -m amazon_photo.prepare --seed 42
```

## Entrenamiento y robustez

Los clasificadores existentes en `amazon_photo/models.py` se conservan: dos capas, ReLU y dropout intermedio. Ambos reciben `model(x, edge_index)`; la MLP ignora las conexiones y la GCN usa `GCNConv` con `cached=False`, por lo que recalcula la normalización al recibir otro grafo.

### Protocolo fijado antes de ejecutar

| Parámetro | Valor predeterminado |
|---|---|
| Partición | Semilla 42; mismas máscaras 60/20/20 en todos los experimentos |
| Semillas de entrenamiento | 42, 43, 44, 45, 46 |
| Capas / dimensión oculta | 2 / 64 |
| Parámetros entrenables | 48.264 por modelo |
| Dropout | 0,5 |
| Optimizador | Adam, learning rate 0,01, weight decay 0,0005 |
| Pérdida | Entropía cruzada, sólo sobre nodos de entrenamiento |
| Selección | Checkpoint con menor pérdida de validación; primer checkpoint en caso de empate |
| Parada | Máximo 300 épocas; detener tras 50 épocas seguidas sin mejora en validación |
| Métricas finales | Accuracy y macro-F1 en train/validación/prueba; también se guarda la pérdida |
| Ejecución | CPU, 1 hilo, algoritmos deterministas de PyTorch |
| Fracción de conexiones eliminadas | 0%, 10%, 25%, 50%, 75%, 90%, 100% |

No se realiza búsqueda de hiperparámetros. La misma configuración se usa en ambos modelos y en todos los niveles de eliminación. La función de entrenamiento recibe únicamente las etiquetas de train y validación; no recibe etiquetas de prueba. No se calculan métricas de prueba por época. Los checkpoints de todas las condiciones de una semilla se fijan antes de evaluar su test.

**Dos preguntas distintas:**

- **`inference` (pesos fijos):** entrenar la GCN con el grafo completo y evaluar ese mismo checkpoint sobre cada grafo incompleto. Mide sensibilidad a conexiones ausentes durante la evaluación.
- **`retrain` (reentrenamiento):** reinicializar y entrenar la GCN en cada grafo incompleto, con sus propias validaciones y parada temprana. Mide qué se puede aprender cuando las conexiones faltan desde el entrenamiento.

La MLP se entrena una sola vez por semilla: sus predicciones no dependen de las aristas. Es la referencia constante en las curvas. El resultado GCN con 0% de eliminación es el mismo checkpoint en ambos protocolos, no dos entrenamientos independientes.

### Cómo se eliminan las conexiones

Para cada semilla se permutan uniformemente los **pares únicos no dirigidos**, sin usar etiquetas ni características. Se eliminan los primeros `floor(fracción × 119081)` pares de esa permutación, retirando ambos sentidos de `edge_index`. Los niveles son anidados: las conexiones eliminadas al 10% también faltan al 25%, etc. La semilla del muestreo es `semilla_entrenamiento + 10000`, mediante un generador local de NumPy que no altera el generador del entrenamiento.

Se mantienen todos los nodos, características y máscaras, incluso si aparecen nuevos aislados. Con 100% de eliminación no quedan conexiones entre productos; `GCNConv` sigue agregando autoenlaces internos para conservar la información propia. Las variantes se construyen en memoria y nunca reemplazan el grafo original.

Las cinco realizaciones combinan variación de inicialización y del muestreo de aristas. Las tablas presentan media y desviación estándar muestral entre semillas, **no intervalos de confianza**. Se usa una sola partición estratificada; estas repeticiones no miden incertidumbre por elección de partición ni generalización a otros datasets. No se garantiza igualdad numérica entre versiones, dispositivos o plataformas; véase la [guía de reproducibilidad de PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html). El macro-F1 da el mismo peso a cada una de las ocho categorías ([definición de Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)).

### Comandos y archivos de salida

```bash
# Sólo comparación con el grafo completo
python -m amazon_photo.experiment --drop-fractions 0

# Experimento completo con las cinco semillas y siete niveles
python -m amazon_photo.experiment

# Otra configuración, conservando los resultados anteriores
python -m amazon_photo.experiment --seeds 42 43 44 --epochs 200 \
  --results-dir results/alternative --artifacts-dir artifacts/alternative
```

`python -m amazon_photo.experiment --help` muestra todas las opciones. Si faltan las máscaras guardadas, ejecutar primero `prepare` con la semilla de partición correspondiente. Cambiar `--seeds` varía el entrenamiento y el muestreo; `--split-seed` selecciona las máscaras previamente generadas.

En `results/experiments/`:

- `manifest.json`: configuración, entorno, huellas, cantidad de parámetros y comprobaciones de conservación de datos.
- `verification.json`: auditoría de la ejecución documentada, incluyendo recarga de checkpoints y recálculo de métricas; no se regenera automáticamente con el CLI.
- `metrics.csv`: resultados finales de cada modelo, protocolo, semilla y fracción eliminada.
- `summary.csv`: medias, desviaciones estándar y cantidad de repeticiones.
- `paired_differences.csv`: diferencia GCN menos MLP en puntos porcentuales, emparejada por semilla.
- `graphs.csv`: conexiones retenidas, fracción realmente eliminada, componentes y aislados por realización.
- `histories/`: pérdidas y accuracy de train/validación por época; sin métricas de test. Las curvas usan `train_loss` y `val_loss` evaluadas sin dropout después de la actualización; `optimization_loss` guarda por separado la pérdida usada para el gradiente, con dropout activo.
- `confusion_matrices.csv`: conteos en prueba con el grafo completo, para la primera semilla solicitada (42 por defecto), elegida de antemano.
- `figures/`: comparación de modelos, convergencia, robustez, fragmentación y matrices de confusión.
- `REPORT.md`: interpretación de la ejecución documentada; no se regenera al cambiar los parámetros del CLI.

Los checkpoints `.pt` se guardan en `artifacts/experiments/`, junto con su configuración, semillas, huellas y época seleccionada. Para cargar uno:

```python
import torch
from amazon_photo.models import build_classifier

checkpoint = torch.load("artifacts/experiments/gcn_seed42_drop0.pt",
                        map_location="cpu", weights_only=True)
model = build_classifier(checkpoint["model"], checkpoint["in_channels"],
                         checkpoint["hidden_channels"], checkpoint["out_channels"],
                         checkpoint["dropout"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Origen y representación

Se utiliza exactamente `torch_geometric.datasets.Amazon(root="data", name="Photo")`, sin `transform` ni `pre_transform`. Según la [documentación oficial de PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Amazon.html), los nodos son productos, las conexiones representan compras conjuntas y las características provienen de reseñas representadas como bolsa de palabras. Las etiquetas corresponden a categorías de producto.

El dataset se distribuye con [gnn-benchmark](https://github.com/shchur/gnn-benchmark) y se describe en [Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/abs/1811.05868). El informe JSON registra la URL exacta utilizada y las huellas SHA-256.

La [carga oficial de PyG](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/amazon.html) lee el NPZ y convierte las conexiones a no dirigidas. En la versión instalada, su lector también binariza las características positivas y quita lazos. Son operaciones propias de la representación estándar de PyG; nuestro código no aplica transformaciones adicionales ni escribe sobre los archivos originales. Se verifica que la caché procesada coincida exactamente con una nueva lectura oficial del NPZ.

`edge_index` almacena cada conexión no dirigida dos veces: `(u, v)` y `(v, u)`. Por eso **238.162 entradas almacenadas equivalen a 119.081 conexiones únicas**. Las estadísticas de aristas únicas usan `u < v`, después de comprobar reciprocidad, ausencia de duplicados y ausencia de lazos. El grado es la cantidad de vecinos únicos; no se divide por dos el grado calculado sobre los orígenes de `edge_index`.

## Datos verificados (etapa 1)

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

El problema es **clasificación transductiva de nodos**: todos los nodos y sus características están disponibles durante el entrenamiento, junto con las aristas de la condición experimental. Sólo las etiquetas de entrenamiento participan en la pérdida y las actualizaciones de parámetros. La MLP usa las mismas características y máscaras. No se generan características a partir de etiquetas.

Se usa `train_test_split` de Scikit-learn dos veces con `stratify` y `random_state=42`: primero 60% entrenamiento y 40% restante, luego mitad validación y mitad prueba sobre ese restante. Las máscaras booleanas tienen forma `(7650,)`, son disjuntas, cubren todos los nodos y mantienen las ocho clases. El redondeo por clase puede variar en uno o dos nodos. Se comprueban sus proporciones, cobertura y reproducibilidad, incluyendo la lectura posterior al guardado.

Las máscaras se guardan en **`data/splits/photo_seed42.npz`**, con claves `train_mask`, `val_mask`, `test_mask`. No se incorporan a la caché original. Ejemplo de carga:

```python
import numpy as np
import torch
from torch_geometric.datasets import Amazon

data = Amazon(root="data", name="Photo")[0]
with np.load("data/splits/photo_seed42.npz", allow_pickle=False) as saved:
    for key in ("train_mask", "val_mask", "test_mask"):
        data[key] = torch.from_numpy(saved[key].copy())
# La pérdida de entrenamiento usa sólo logits[data.train_mask]
# y data.y[data.train_mask].
```

Las etiquetas de validación sirven exclusivamente para evaluación y selección de hiperparámetros o parada temprana, sin gradientes. Las etiquetas de prueba se reservan para la evaluación final y no pueden orientar entrenamiento ni selección. La estratificación y las estadísticas descriptivas solicitadas —como la homofilia global— sí consultan etiquetas, pero no crean entradas del modelo ni definen sus parámetros. No se debe usar este análisis de etiquetas de prueba para elegir una arquitectura o ajustar hiperparámetros.

Las máscaras permanecen fijas al comparar MLP/GCN y los niveles de eliminación de conexiones.

## Estructura

```text
amazon_photo/
  dataset.py         # Carga, huellas e integridad
  splits.py          # Creación y validación de máscaras
  analysis.py        # Estadísticas y tablas
  plots.py           # Cuatro figuras con Matplotlib
  prepare.py         # CLI que coordina la preparación
  models.py          # MLP y GCN de dos capas (etapa 2)
  training.py        # Pérdida en train y selección por validación
  perturbation.py    # Eliminación anidada de pares no dirigidos
  experiment.py      # CLI de comparación y robustez
  experiment_plots.py
tests/
  test_preparation.py
  test_models.py
  test_experiments.py
data/                # Ignorado por Git; caché PyG y máscaras regenerables
artifacts/           # Ignorado por Git; checkpoints regenerables
results/
  dataset_report.json
  figures/           # Figuras PNG del análisis real
  tables/            # CSV de categorías, grados y componentes
  experiments/       # Informe, métricas, historias y figuras de la etapa 3
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
- **18 pruebas automatizadas superadas**: preparación, comportamiento de MLP/GCN, actualización independiente de etiquetas ajenas a train, selección y restauración del checkpoint por validación, repetibilidad, macro-F1 y eliminación simétrica/anidada sin modificar el grafo original.
- `pip check`: **sin dependencias incompatibles**.
- Una **descarga nueva** desde GitHub y su procesamiento con PyG produjeron los mismos tensores que la copia local. Las huellas están registradas en [`results/dataset_report.json`](results/dataset_report.json).
- Dos ejecuciones con semilla 42 produjeron las mismas huellas de los tensores y del contenido de las máscaras. Las cuatro figuras se revisaron visualmente.
- La etapa 3 completó 40 entrenamientos y 75 condiciones de evaluación: se recargaron todos los checkpoints y se recalcularon sus métricas. Una repetición independiente de la semilla 42 reprodujo exactamente las historias y resultados con el grafo completo. Véase [`results/experiments/verification.json`](results/experiments/verification.json).

## Limitaciones

- Se reutilizaron las dependencias ya instaladas; no se comprobó una instalación desde cero ni la compatibilidad con otras plataformas.
- PyTorch emite un `FutureWarning` sobre `torch.jit.script` con Python 3.14. Esta etapa no utiliza JIT y terminó correctamente; su compatibilidad deberá revisarse antes de usar compilación.
- El NPZ original permanece intacto. La representación cargada corresponde al procesamiento estándar de PyG explicado arriba.
- La URL de origen apunta a `master`; las huellas permiten detectar cambios futuros. El hash binario de `data.pt` puede variar entre versiones de serialización aunque sus tensores coincidan.
- El estudio usa Amazon Photo, una partición fija y eliminación uniforme de conexiones. No permite concluir lo mismo para conexiones ausentes de forma dirigida o para otros datasets.
