# FactoMineR en Python: Análisis Exploratorio Multivariante 📊

Este repositorio contiene una implementación en **Python** de la metodología clásica de **FactoMineR** (la librería estándar en R para Análisis Exploratorio de Datos), junto con un dashboard interactivo profesional.

El objetivo es cerrar la brecha entre la estadística académica (R/FactoMineR) y el ecosistema de producción (Python/Streamlit).

## 📂 Contenido del Repositorio

### 1. `02_PCA_FactoMineR_style.py` (Script de Análisis)

Un script de Python puro que replica paso a paso el output de la función `PCA()` de FactoMineR.

- **Librería principal**: `prince`, `pandas`, `matplotlib`.
- **Salida**: Genera gráficos estáticos (`.png`) y un reporte en Excel (`.xlsx`) idéntico al de R.
- **Métricas**: Calcula autovalores, coordenadas, contribuciones y Cos2 tanto para individuos como para variables.

### 2. `03_Dashboard_PCA.py` (Dashboard Interactivo)

Una aplicación web interactiva construida con **Streamlit** y **Plotly**.

- **Visualización Dinámica**: Scree plots interactivos, mapas de individuos y círculos de correlación.
- **Robustez**: Manejo automático de errores de conexión y selección inteligente de variables numéricas.
- **Estilo**: Interfaz moderna y responsiva lista para presentaciones.

### 3. `Explicacion_02_PCA_FactoMineR_style.md`

Documentación técnica detallada que explica la matemática y la lógica detrás del código, diferenciando entre variables activas y suplementarias.

---

## 🚀 Instalación y Uso

### Prerrequisitos

Instala las dependencias necesarias:

```bash
pip install pandas numpy matplotlib seaborn prince openpyxl requests streamlit plotly
```

### Ejecutar el Análisis (Script)

Para generar los reportes estáticos y el Excel:

```bash
python 02_PCA_FactoMineR_style.py
```

### Ejecutar el Dashboard

Para lanzar la aplicación web:

```bash
streamlit run 03_Dashboard_PCA.py
```

---

## 📚 Referencias Académicas

Este proyecto sigue la metodología enseñada en:

- **Husson, F., Le, S., & Pages, J. (2017)**. _Exploratory Multivariate Analysis by Example Using R_. CRC Press.
- **Curso FactoMineR**: [http://factominer.free.fr/](http://factominer.free.fr/)

---

**Autor**: [Tu Nombre / @alxz0212]
**Profesor/Referencia**: Juan Marcelo Gutiérrez Miranda (@TodoEconometria)
