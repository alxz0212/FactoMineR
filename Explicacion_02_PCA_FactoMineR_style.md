# Explicación Detallada: Análisis de Componentes Principales (PCA) al Estilo FactoMineR

Este documento explica en detalle el funcionamiento del script `02_PCA_FactoMineR_style.py`, diseñado para **replicar la metodología de FactoMineR** (el estándar de oro en R para análisis exploratorio) utilizando Python y la librería `prince`.

## 📌 1. ¿Por qué "Estilo FactoMineR"?

La mayoría de tutoriales de PCA en Python usan `scikit-learn`, que está optimizado para **Machine Learning** (predicción y reducción de dimensionalidad antes de un modelo).

Sin embargo, **FactoMineR** se enfoca en la **Estadística Exploratoria Multivariante**, cuyo objetivo es _entender_ los datos:

- ¿Qué variables están correlacionadas?
- ¿Qué individuos se parecen entre sí?
- ¿Qué variables caracterizan a los grupos de individuos?
- ¿Qué calidad tiene la representación gráfica?

Este script utiliza la librería `prince` porque, a diferencia de `scikit-learn`, ofrece de forma nativa las estadísticas clásicas de FactoMineR: **Inercia (Autovalores), Coordenadas, Contribuciones y Cos2**.

---

## 🛠️ 2. Librerías y Requisitos

### Instalación

Para ejecutar este script, es necesario instalar las siguientes dependencias. Puedes hacerlo con un solo comando:

```bash
pip install pandas numpy matplotlib seaborn prince openpyxl requests
```

### Librerías Utilizadas

- **Prince**: El motor principal. Es esencial porque calcula las estadísticas detalladas (contribuciones, cosenos cuadrados) que `scikit-learn` no entrega por defecto.
- **Pandas**: Manejo de estructuras de datos (DataFrames) y lectura/escritura de archivos (CSV, Excel).
- **NumPy**: Cálculos numéricos y matriciales de base.
- **Matplotlib y Seaborn**: Motores gráficos para construir las visualizaciones personalizadas (biplots, scree plots).
- **OpenPyXL**: Motor para exportar los resultados finales a archivos Excel `.xlsx`.

## 📂 3. El Dataset: Decathlon

El script utiliza el clásico dataset **Decathlon** (resultados de atletas en 10 pruebas olímpicas).

- **Filas (Individuos)**: Atletas (Sebrle, Clay, Karpov, etc.).
- **Columnas (Variables)**: Tiempos en 100m, longitud de salto, lanzamiento de peso, etc.

---

## 🔑 3. Concepto Clave: Variables Activas vs. Suplementarias

Esta es una de las distinciones más importantes en la metodología FactoMineR que este script implementa manualmente:

### A. Variables Activas

Son las que **construyen los ejes principales**. El PCA se entrena _solo_ con ellas.

- En el script: Las 10 pruebas deportivas (`100m`, `Long.jump`, `Shot.put`, etc.).
- La "nube de puntos" se forma basándose únicamente en el rendimiento deportivo.

### B. Variables Suplementarias (Ilustrativas)

No influyen en la construcción de los ejes, pero se **proyectan** sobre ellos a posteriori para ayudar a la interpretación.

- **Cuantitativas**: `Rank` (Ranking final), `Points` (Puntos totales). Nos permiten ver, por ejemplo, si el Eje 1 está correlacionado con ganar más puntos.
- **Cualitativas**: `Competition` (Juegos Olímpicos vs Decastar). Nos permite colorear los individuos para ver si hay diferencias de rendimiento según la competición.

```python
# Variables ACTIVAS (construyen ejes)
active_cols = df.columns[:10].tolist()

# Variables SUPLEMENTARIAS (solo se proyectan)
sup_cols_quanti = ['Rank', 'Points']
sup_cols_quali = ['Competition']
```

---

## ⚙️ 4. Configuración del Modelo PCA

El script configura `PCA` de `prince` para comportarse exactamente como R:

```python
pca = PCA(
    n_components=5,           # Número de dimensiones a analizar
    rescale_with_mean=True,   # Centrar datos (media = 0)
    rescale_with_std=True,    # Escalar datos (std = 1) -> EQUIVALE A scale.unit=TRUE en R
    ...
)
```

- **`rescale_with_std=True`**: Es fundamental. Significa que hacemos un **PCA Normado**. Al dividir cada variable por su desviación estándar, evitamos que variables con unidades grandes (ej. `Points` ~8000) dominen sobre variables pequeñas (ej. `High.jump` ~2.0). Todas las variables tienen el mismo peso inicial.

---

## 📊 5. Interpretación de Resultados (El "Output" de FactoMineR)

El script calcula y muestra 4 métricas fundamentales para cada dimensión (Eje):

### 1. Autovalores (Eigenvalues)

Indican cuánta información (inercia/varianza) retiene cada eje.

- **Regla de Kaiser**: Se suelen retener los ejes con autovalor > 1 (explican más que una sola variable original promedio).

### 2. Coordenadas (Factor Scores)

Son las nuevas "direcciones" de los individuos en el mapa.

- Ejemplo: Si un atleta tiene un valor muy alto en el Eje 1 (positivo) y el Eje 1 representa "Fuerza", ese atleta es muy fuerte.

### 3. Contribuciones (Contributions)

Indican **quién construyó el eje**. La suma de contribuciones para un eje es 100%.

- **Individuos**: ¿Qué atletas son los extremos que definen la dimensión? (Ej. "Sebrle" define el extremo positivo).
- **Variables**: ¿Qué pruebas pesan más en la dimensión? (Ej. Si `100m` y `110m.hurdle` contribuyen mucho al Eje 2, ese eje es "Velocidad").
- _Ayuda a ponerle nombre a los ejes._

### 4. Cos2 (Coseno Cuadrado - Calidad de Representación)

Mide qué tan bien se ve un individuo o variable en el mapa 2D actual.

- Valor entre 0 y 1.
- **Cercano a 1**: El punto está muy cerca del plano proyectado. Lo que vemos en el gráfico es real.
- **Cercano a 0**: El punto está lejos del plano (quizás se explica mejor en el Eje 3 o 4). **Cuidado al interpretar estos puntos en el gráfico**, su posición puede ser engañosa por la perspectiva.

---

## 🎨 6. Gráficos Generados

El script genera una imagen compuesta (`02_PCA_FactoMineR_graficos.png`) con 6 paneles:

1.  **Scree Plot**: Gráfico de barras de la varianza explicada. Busca el "codo" donde la ganancia de información se aplana.
2.  **Círculo de Correlación**:
    - Muestra las relaciones entre variables.
    - Ángulo agudo (< 90°): Correlación positiva.
    - Ángulo obtuso (> 90°): Correlación negativa (ej. Tiempo en 100m vs Puntos: a más tiempo, menos puntos).
    - Ángulo recto (90°): Sin correlación.
    - **Longitud de la flecha**: Calidad de representación (Cos2). Flechas cortas = mala representación.
3.  **Mapa de Individuos**: La "nube de puntos" de los atletas.
    - Coloreado por la variable suplementaria `Competition`.
4.  **Contribuciones (Barplots)**: Para Dimensión 1 y 2. Las barras rojas indican variables que contribuyen más de la media (las más importantes para definir ese eje).
5.  **Biplot**: Superposición de individuos y flechas de variables. Útil para ver "tendencias". (Ej. Atletas en la dirección de la flecha `Javelin` son buenos en jabalina).

---

## 💾 7. Exportación a Excel

Finalmente, el script emula la salida tabular completa guardando todo en `02_PCA_resultados_FactoMineR.xlsx`. Esto es ideal para informes, ya que permite explorar:

- Datos exactos de cada atleta.
- Correlaciones precisas.
- Filtros por calidad (Cos2) antes de interpretar.

---

## Resumen del Flujo de Trabajo

1.  **Instalar**: `pip install prince pandas matplotlib seaborn`
2.  **Cargar**: Tus datos numéricos.
3.  **Separar**: Define qué columnas son activas (para el cálculo) y cuáles ilustrativas.
4.  **Ejecutar Script**: Obtendrás los gráficos y el Excel.
5.  **Interpretar**:
    - Mira el _Scree Plot_ para decidir cuántos ejes valen la pena.
    - Usa las _Contribuciones_ para nombrar los ejes (ej. "Eje 1: Potencia", "Eje 2: Velocidad").
    - Usa el _Mapa de Individuos_ para ver clusters y outliers.
    - Usa el _Círculo de Correlación_ para entender relaciones entre variables.

Este script es una plantilla robusta para realizar Análisis Exploratorio de Datos (EDA) serio y académico en Python.
