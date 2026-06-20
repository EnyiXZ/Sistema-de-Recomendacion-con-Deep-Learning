# Sistema de Recomendación Personalizado con Deep Learning

Este repositorio contiene el desarrollo completo de un **Sistema de Recomendación Inteligente** diseñado bajo un enfoque de aprendizaje supervisado profundo (Deep Learning). El modelo predice la probabilidad de que un usuario adquiera un producto específico basándose en su historial de interacciones y en atributos contextuales (categoría, estilo, temporada y tendencias), permitiendo tanto la generación de recomendaciones personalizadas como la resolución del problema de inicio en frío (*Cold Start*).

## 📋 Objetivo del Proyecto

El objetivo principal es construir y entrenar una red neuronal profunda en TensorFlow/Keras utilizando capas de **Embeddings** para mapear variables categóricas (usuarios, productos y contextos) en espacios vectoriales continuos, capturando relaciones complejas de afinidad para optimizar las tasas de conversión (conversión de clicks a compras).

## 📊 Estructura y Análisis del Dataset

El proyecto trabaja sobre una única tabla estructurada con **5,000 interacciones históricas** entre **200 usuarios únicos** y un catálogo de **50 productos únicos**. 

### Atributos del Dataset:

| Columna | Tipo | Descripción |
| :--- | :--- | :--- |
| **USER_ID** | Int (Cat) | Identificador único del cliente (200 usuarios). |
| **ITEM_ID** | Int (Cat) | Identificador único del producto (50 ítems). |
| **CATEGORY** | String | Tipo de prenda (shoes, tshirt, jacket, dress, bag, etc.). |
| **STYLE** | String | Enfoque de la prenda (casual, elegant, sport, urban). |
| **SEASON** | String | Estacionalidad óptima (summer, winter, spring, autumn). |
| **IS_TRENDING**| Binario | Indica si el producto es tendencia actual (1) o no (0). |
| **PURCHASE** | Binario | **Target (Variable Objetivo)**: 1 si se efectuó la compra, 0 si no. |

### Insights Clave del EDA:
* **Desbalanceo de Clases:** El dataset presenta un sesgo hacia la no-compra, con un **ratio de conversión general del 25.4%** (3,728 instancias de no-compra frente a 1,272 de compra).
* **Impacto de Tendencias:** Los productos marcados como *Trending* exhiben un ratio de compra significativamente mayor (36.9%) en comparación con los no-trending (20.5%).
* **Densidad por Usuario:** Los usuarios promedian 25 interacciones dentro del histórico, con un rango controlado de entre 16 como mínimo y 41 como máximo.

## 🧠 Arquitectura del Modelo

Para procesar la naturaleza dispersa de los IDs de usuario y artículo junto con sus metadatos, se diseñó una arquitectura de red híbrida utilizando la API funcional de Keras:

1. **Capas de Embedding independientes:**
   * `USER_ID` $\rightarrow$ Vectores densos de dimensión 16.
   * `ITEM_ID` $\rightarrow$ Vectores densos de dimensión 16.
   * `CATEGORY` $\rightarrow$ Vectores densos de dimensión 8.
   * `STYLE` $\rightarrow$ Vectores densos de dimensión 8.
   * `SEASON` $\rightarrow$ Vectores densos de dimensión 4.
2. **Fusión de Características:** Extracción y aplanado (`Flatten`) de las salidas de los embeddings para su posterior concatenación (`Concatenate`) junto con la señal binaria de `IS_TRENDING`.
3. **Bloques Densos Avanzados:** 
   * Capa totalmente conectada de 128 neuronas + Normalización por Lote (`BatchNormalization`) + `Dropout(0.3)`.
   * Capa totalmente conectada de 64 neuronas + `BatchNormalization` + `Dropout(0.2)`.
   * Capa totalmente conectada de 32 neuronas con activación ReLU.
4. **Capa de Salida:** Una única neurona con función de activación `Sigmoid` para retornar un valor probabilístico continuo en el rango $[0, 1]$.

## ⚙️ Estrategia de Entrenamiento y Regularización

Para contrarrestar el desbalanceo de la variable objetivo y garantizar la estabilidad del entrenamiento se implementaron las siguientes técnicas:
* **Pesos de Clase Suavizados:** Aplicación de penalizaciones personalizadas en la función de pérdida calculadas mediante la raíz cuadrada de la proporción inversa de las clases (`{0: 1.0, 1: 1.7115}`).
* **Mecanismos de Control (Callbacks):**
  * `EarlyStopping`: Interrupción del entrenamiento tras 8 épocas consecutivas sin mejoras en la pérdida de validación (`val_loss`), restaurando los mejores pesos.
  * `ReduceLROnPlateau`: Reducción del factor de aprendizaje por un coeficiente de 0.2 ante el estancamiento de la convergencia.

## 📈 Resultados de Rendimiento

Tras la convergencia y detención temprana en la época 13, el modelo obtuvo las siguientes métricas de rendimiento sobre el conjunto de test independiente (20% de los datos):

* **Loss (Binary Crossentropy):** 0.5758
* **Accuracy General:** 72.60%
* **Área Bajo la Curva (AUC - ROC):** 0.6058

## 🎯 Capacidades del Motor de Recomendación

El sistema cuenta con funciones avanzadas integradas listas para producción:
1. **Recomendación Personalizada:** Dado un `USER_ID` activo, el sistema genera dinámicamente predicciones sintéticas cruzando al usuario con todo el catálogo de productos disponibles bajo el contexto actual del mercado, ordenando y devolviendo los Top-K productos con mayor probabilidad de compra.
2. **Mitigación de Inicio en Frío (Cold Start):** En caso de ingresar un usuario completamente nuevo sin historial en la matriz de embeddings, el sistema activa un método heurístico de respaldo que recomienda los productos con mayor rendimiento global basándose puramente en las combinaciones óptimas de categorías y tendencias vigentes.

## 🛠️ Requisitos del Sistema

* Python 3.10+
* TensorFlow 2.15+ / 2.19
* Scikit-Learn
* Pandas & NumPy
* Matplotlib & Seaborn
