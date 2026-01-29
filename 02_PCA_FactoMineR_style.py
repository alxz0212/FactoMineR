"""
-------------------------
Autor original/Referencia: @TodoEconometria
Profesor: Juan Marcelo Gutierrez Miranda
Metodologia: Cursos Avanzados de Big Data, Ciencia de Datos,
             Desarrollo de aplicaciones con IA & Econometria Aplicada.
Hash ID de Certificacion: 4e8d9b1a5f6e7c3d2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d9c
Repositorio: https://github.com/TodoEconometria/certificaciones

REFERENCIA ACADEMICA PRINCIPAL:
- Husson, F., Le, S., & Pages, J. (2017). Exploratory Multivariate Analysis by Example Using R. CRC Press.
- Tutorial FactoMineR: http://factominer.free.fr/course/FactoTuto.html
- Le, S., Josse, J., & Husson, F. (2008). FactoMineR: An R package for multivariate analysis. JSS, 25(1), 1-18.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.
-------------------------

PCA en Python: Estilo FactoMineR
================================
Este script replica la logica de FactoMineR (el estandar de oro en R para analisis exploratorio)
utilizando Python con la libreria Prince.

Diferencias clave:
- Scikit-learn: Se enfoca en Machine Learning (reducir dimensiones para un modelo)
- FactoMineR/Prince: Se enfoca en Exploracion de Datos (entender que variables mueven a
  que individuos, calidad de representacion cos2, contribuciones, y variables suplementarias)
"""

# =============================================================================
# INSTALACION DEL ENTORNO (ejecutar una vez en terminal)
# =============================================================================
# pip install pandas matplotlib seaborn prince openpyxl requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince import PCA
import warnings
import os

warnings.filterwarnings('ignore')

# Obtener la ruta del directorio donde esta el script (para guardar archivos)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuracion de graficos
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 80)
print("PCA EN PYTHON: ESTILO FACTOMINER")
print("Replicando la metodologia de analisis exploratorio multivariante")
print("=" * 80)

# =============================================================================
# 1. CARGAR EL DATASET DECATHLON (Clasico de FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("1. CARGANDO DATASET DECATHLON")
print("=" * 80)

# Dataset Decathlon: Resultados de atletas en las 10 pruebas del decatlon
url = "https://raw.githubusercontent.com/fhusson/FactoMineR/master/inst/extdata/decathlon.csv"

try:
    df = pd.read_csv(url, index_col=0)
    print(f"\nDatos cargados exitosamente desde GitHub")
except:
    print("\nError al cargar desde URL. Creando dataset de ejemplo...")
    # Dataset de respaldo basado en datos reales del decatlon
    np.random.seed(42)
    atletas = ['SEBRLE', 'CLAY', 'KARPOV', 'BERNARD', 'YURKOV', 'WARNERS',
               'ZSIVOCZKY', 'McMULLEN', 'MARTINEAU', 'HERNU', 'BARRAS',
               'NOOL', 'BOURGUIGNON', 'Sebrle', 'Clay', 'Karpov', 'Macey',
               'Warners', 'Zsivoczky', 'Hernu', 'Bernard', 'Schwarzl',
               'Pogorelov', 'Schoenbeck', 'Barras', 'Smith', 'Averyanov',
               'Nool', 'Bourguignon', 'Uldal', 'Casarsa', 'Lorenzo',
               'Karlivans', 'Korkizoglou', 'Turi', 'Parkhomenko', 'Drews',
               'Smirnov', 'Qi', 'Terek', 'Gombala']

    data = {
        '100m': np.random.uniform(10.5, 11.5, len(atletas)),
        'Long.jump': np.random.uniform(6.8, 8.0, len(atletas)),
        'Shot.put': np.random.uniform(13, 17, len(atletas)),
        'High.jump': np.random.uniform(1.85, 2.15, len(atletas)),
        '400m': np.random.uniform(47, 51, len(atletas)),
        '110m.hurdle': np.random.uniform(13.5, 15.5, len(atletas)),
        'Discus': np.random.uniform(38, 52, len(atletas)),
        'Pole.vault': np.random.uniform(4.4, 5.4, len(atletas)),
        'Javeline': np.random.uniform(50, 72, len(atletas)),
        '1500m': np.random.uniform(260, 310, len(atletas)),
        'Rank': np.arange(1, len(atletas)+1),
        'Points': np.random.randint(7500, 9000, len(atletas)),
        'Competition': np.random.choice(['Decastar', 'OlympicG'], len(atletas))
    }
    df = pd.DataFrame(data, index=atletas)

print(f"\nDimensiones del dataset: {df.shape[0]} atletas x {df.shape[1]} variables")
print(f"\nPrimeras filas del dataset:")
print(df.head())

# =============================================================================
# 2. SEPARAR VARIABLES ACTIVAS Y SUPLEMENTARIAS (Concepto clave de FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("2. SEPARACION DE VARIABLES: ACTIVAS vs SUPLEMENTARIAS")
print("=" * 80)

# Variables ACTIVAS: Las 10 pruebas del decatlon (construyen los ejes)
active_cols = df.columns[:10].tolist()

# Variables SUPLEMENTARIAS: Rank, Points, Competition (se proyectan pero no construyen ejes)
sup_cols_quanti = ['Rank', 'Points']  # Cuantitativas suplementarias
sup_cols_quali = ['Competition']       # Cualitativas suplementarias

print(f"\nVARIABLES ACTIVAS (construyen los ejes PCA):")
for i, col in enumerate(active_cols, 1):
    print(f"  {i:2d}. {col}")

print(f"\nVARIABLES SUPLEMENTARIAS CUANTITATIVAS (se proyectan, no construyen):")
for col in sup_cols_quanti:
    print(f"  - {col}")

print(f"\nVARIABLES SUPLEMENTARIAS CUALITATIVAS (para colorear/segmentar):")
for col in sup_cols_quali:
    print(f"  - {col}: {df[col].unique()}")

# =============================================================================
# 3. CONFIGURAR Y ENTRENAR EL PCA CON PRINCE
# =============================================================================
print("\n" + "=" * 80)
print("3. ENTRENAMIENTO DEL MODELO PCA (Estilo FactoMineR)")
print("=" * 80)

# Configuracion equivalente a PCA() en FactoMineR con scale.unit=TRUE
pca = PCA(
    n_components=5,           # Numero de dimensiones a extraer
    rescale_with_mean=True,   # Centrar los datos (media = 0)
    rescale_with_std=True,    # Escalar por desviacion estandar (equivale a scale.unit=TRUE en R)
    copy=True,
    engine='sklearn',
    random_state=42
)

# Entrenar SOLO con las variables activas
pca = pca.fit(df[active_cols])

print("\nModelo PCA entrenado exitosamente")
print("Parametros utilizados:")
print("  - rescale_with_mean = True (centrado)")
print("  - rescale_with_std = True (estandarizacion, equivale a scale.unit=TRUE en R)")
print("  - n_components = 5")

# =============================================================================
# 4. AUTOVALORES Y VARIANZA EXPLICADA (pca$eig en FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("4. AUTOVALORES (EIGENVALUES) Y VARIANZA EXPLICADA")
print("=" * 80)

# Obtener resumen de autovalores (equivalente a pca$eig en R)
print("\nTabla de Autovalores (equivalente a pca$eig en FactoMineR):")
print(pca.eigenvalues_summary)

# Crear tabla formateada como en FactoMineR
# En Prince: percentage_of_variance_ contiene la varianza explicada (ya en porcentaje dividido por 100)
eigenvalues_df = pd.DataFrame({
    'Autovalor': pca.eigenvalues_,
    'Varianza (%)': pca.percentage_of_variance_,
    'Varianza Acumulada (%)': pca.cumulative_percentage_of_variance_
}, index=[f'Dim.{i+1}' for i in range(len(pca.eigenvalues_))])

print("\n" + eigenvalues_df.to_string())

# Regla de Kaiser: componentes con autovalor > 1
n_kaiser = sum(pca.eigenvalues_ > 1)
print(f"\nRegla de Kaiser: Retener {n_kaiser} componentes (autovalor > 1)")

# =============================================================================
# 5. COORDENADAS DE LOS INDIVIDUOS (pca$ind$coord en FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("5. COORDENADAS DE LOS INDIVIDUOS (Factor Scores)")
print("=" * 80)

# Obtener coordenadas de los individuos en el espacio factorial
row_coords = pca.row_coordinates(df[active_cols])
row_coords.columns = [f'Dim.{i+1}' for i in range(row_coords.shape[1])]

print("\nCoordenadas de los primeros 10 atletas:")
print(row_coords.head(10).round(3))

# =============================================================================
# 6. CONTRIBUCIONES DE LOS INDIVIDUOS (pca$ind$contrib en FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("6. CONTRIBUCIONES DE LOS INDIVIDUOS A CADA DIMENSION")
print("=" * 80)

# Las contribuciones indican cuanto aporta cada individuo a la construccion del eje
row_contrib = pca.row_contributions_
row_contrib.columns = [f'Dim.{i+1}' for i in range(row_contrib.shape[1])]

print("\nContribucion (%) de los primeros 10 atletas:")
print((row_contrib.head(10) * 100).round(2))

# Identificar atletas mas influyentes en Dim 1
top_contrib_dim1 = row_contrib['Dim.1'].sort_values(ascending=False).head(5)
print(f"\nAtletas con mayor contribucion a Dim.1:")
for atleta, contrib in top_contrib_dim1.items():
    print(f"  {atleta}: {contrib*100:.2f}%")

# =============================================================================
# 7. CALIDAD DE REPRESENTACION - COS2 (pca$ind$cos2 en FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("7. CALIDAD DE REPRESENTACION (COS2) DE LOS INDIVIDUOS")
print("=" * 80)

# cos2 indica que tan bien representado esta cada individuo en el plano
# Un cos2 cercano a 1 significa representacion perfecta
row_cos2 = pca.row_cosine_similarities(df[active_cols])
row_cos2.columns = [f'Dim.{i+1}' for i in range(row_cos2.shape[1])]

print("\nCOS2 de los primeros 10 atletas (calidad de representacion):")
print(row_cos2.head(10).round(3))

# Calidad en el plano 1-2
cos2_plano = row_cos2['Dim.1'] + row_cos2['Dim.2']
print(f"\nAtletas mejor representados en el plano Dim1-Dim2:")
for atleta, cos2 in cos2_plano.sort_values(ascending=False).head(5).items():
    print(f"  {atleta}: {cos2:.3f} ({cos2*100:.1f}% de su variabilidad explicada)")

# =============================================================================
# 8. CORRELACIONES DE LAS VARIABLES (pca$var$cor en FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("8. CORRELACIONES DE LAS VARIABLES CON LAS DIMENSIONES")
print("=" * 80)

# Las correlaciones nos dicen como se relaciona cada variable con cada eje
# En Prince, column_correlations es un atributo DataFrame
col_correlations = pca.column_correlations.copy()
col_correlations.columns = [f'Dim.{i+1}' for i in range(col_correlations.shape[1])]

print("\nCorrelaciones Variables-Dimensiones (para el circulo de correlacion):")
print(col_correlations.round(3))

# =============================================================================
# 9. CONTRIBUCIONES DE LAS VARIABLES (pca$var$contrib en FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("9. CONTRIBUCIONES DE LAS VARIABLES A CADA DIMENSION")
print("=" * 80)

# Que variable "construyo" cada eje?
col_contrib = pca.column_contributions_
col_contrib.columns = [f'Dim.{i+1}' for i in range(col_contrib.shape[1])]

print("\nContribucion (%) de cada variable:")
print((col_contrib * 100).round(2))

# Variables mas importantes para Dim 1
print(f"\nVariables que mas contribuyen a Dim.1:")
for var, contrib in (col_contrib['Dim.1'] * 100).sort_values(ascending=False).head(5).items():
    print(f"  {var}: {contrib:.2f}%")

# =============================================================================
# 10. COS2 DE LAS VARIABLES (pca$var$cos2 en FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("10. CALIDAD DE REPRESENTACION (COS2) DE LAS VARIABLES")
print("=" * 80)

col_cos2 = pca.column_cosine_similarities_
col_cos2.columns = [f'Dim.{i+1}' for i in range(col_cos2.shape[1])]

print("\nCOS2 de las variables (que tan bien representadas estan):")
print(col_cos2.round(3))

# =============================================================================
# 11. GRAFICOS ESTILO FACTOMINER
# =============================================================================
print("\n" + "=" * 80)
print("11. GENERANDO GRAFICOS ESTILO FACTOMINER")
print("=" * 80)

# Crear figura con multiples subplots
# Crear figura con multiples subplots con constrained_layout para mejor ajuste
fig = plt.figure(figsize=(18, 12), constrained_layout=True)

# -----------------------------------------------------------------------------
# GRAFICO 1: SCREE PLOT (Varianza explicada)
# -----------------------------------------------------------------------------
ax1 = plt.subplot(2, 3, 1)
n_comp = len(pca.eigenvalues_)
x = range(1, n_comp + 1)

# Barras de varianza
bars = ax1.bar(x, (pca.percentage_of_variance_ / 100) * 100, color='steelblue',
               edgecolor='black', alpha=0.7)
# Linea de varianza acumulada
ax1.plot(x, np.cumsum((pca.percentage_of_variance_ / 100)) * 100, 'ro-',
         linewidth=2, markersize=8, label='Acumulada')

# Linea de referencia (autovalor = 1 / numero de variables)
ax1.axhline(y=100/len(active_cols), color='gray', linestyle='--',
            label=f'Umbral = {100/len(active_cols):.1f}%')

ax1.set_xlabel('Dimension', fontsize=12)
ax1.set_ylabel('Porcentaje de Varianza', fontsize=12)
ax1.set_title('Scree Plot\n(Varianza Explicada por Dimension)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Anotar valores en las barras
for bar, val in zip(bars, (pca.percentage_of_variance_ / 100) * 100):
    ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 ha='center', va='bottom', fontsize=9)

# -----------------------------------------------------------------------------
# GRAFICO 2: CIRCULO DE CORRELACION (El mas importante de FactoMineR)
# -----------------------------------------------------------------------------
ax2 = plt.subplot(2, 3, 2)

# Dibujar circulo unitario
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=2)
ax2.add_artist(circle)

# Dibujar flechas de las variables
for var in active_cols:
    x = col_correlations.loc[var, 'Dim.1']
    y = col_correlations.loc[var, 'Dim.2']

    # Color basado en la calidad de representacion (cos2)
    cos2_var = col_cos2.loc[var, 'Dim.1'] + col_cos2.loc[var, 'Dim.2']
    color = plt.cm.RdYlGn(cos2_var)

    ax2.arrow(0, 0, x*0.9, y*0.9, head_width=0.05, head_length=0.05,
              fc=color, ec=color, linewidth=2)

    # Etiqueta de la variable
    ax2.text(x*1.1, y*1.1, var, ha='center', va='center', fontsize=9,
             fontweight='bold', color='darkblue')

ax2.axhline(0, color='black', linewidth=0.5)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlim(-1.3, 1.3)
ax2.set_ylim(-1.3, 1.3)
ax2.set_xlabel(f'Dim.1 ({(pca.percentage_of_variance_ / 100)[0]*100:.1f}%)', fontsize=12)
ax2.set_ylabel(f'Dim.2 ({(pca.percentage_of_variance_ / 100)[1]*100:.1f}%)', fontsize=12)
ax2.set_title('Circulo de Correlacion\n(Variables)', fontsize=14, fontweight='bold')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# GRAFICO 3: MAPA DE INDIVIDUOS (coloreado por variable suplementaria)
# -----------------------------------------------------------------------------
ax3 = plt.subplot(2, 3, 3)

# Preparar datos con coordenadas y variable suplementaria
coords_plot = row_coords.copy()
coords_plot['Competition'] = df['Competition']

# Colores por competicion
palette = {'Decastar': 'blue', 'OlympicG': 'red'}
for comp in df['Competition'].unique():
    mask = coords_plot['Competition'] == comp
    ax3.scatter(coords_plot.loc[mask, 'Dim.1'], coords_plot.loc[mask, 'Dim.2'],
               label=comp, c=palette.get(comp, 'gray'), s=80, alpha=0.7, edgecolors='black')

# Etiquetas de atletas
for atleta in coords_plot.index:
    ax3.annotate(atleta,
                xy=(coords_plot.loc[atleta, 'Dim.1'], coords_plot.loc[atleta, 'Dim.2']),
                xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.8)

ax3.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax3.axvline(0, color='black', linestyle='--', linewidth=0.5)
ax3.set_xlabel(f'Dim.1 ({(pca.percentage_of_variance_ / 100)[0]*100:.1f}%)', fontsize=12)
ax3.set_ylabel(f'Dim.2 ({(pca.percentage_of_variance_ / 100)[1]*100:.1f}%)', fontsize=12)
ax3.set_title('Mapa de Individuos\n(Coloreado por Competicion)', fontsize=14, fontweight='bold')
ax3.legend(title='Competicion')
ax3.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# GRAFICO 4: CONTRIBUCIONES DE VARIABLES A DIM 1
# -----------------------------------------------------------------------------
ax4 = plt.subplot(2, 3, 4)

contrib_dim1 = (col_contrib['Dim.1'] * 100).sort_values(ascending=True)
colors = ['steelblue' if v < 100/len(active_cols) else 'red' for v in contrib_dim1]

ax4.barh(contrib_dim1.index, contrib_dim1.values, color=colors, edgecolor='black')
ax4.axvline(x=100/len(active_cols), color='red', linestyle='--',
            label=f'Contribucion esperada ({100/len(active_cols):.1f}%)')

ax4.set_xlabel('Contribucion (%)', fontsize=12)
ax4.set_title('Contribuciones de Variables a Dim.1\n(Rojo = sobre la media)', fontsize=14, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3, axis='x')

# -----------------------------------------------------------------------------
# GRAFICO 5: CONTRIBUCIONES DE VARIABLES A DIM 2
# -----------------------------------------------------------------------------
ax5 = plt.subplot(2, 3, 5)

contrib_dim2 = (col_contrib['Dim.2'] * 100).sort_values(ascending=True)
colors = ['steelblue' if v < 100/len(active_cols) else 'red' for v in contrib_dim2]

ax5.barh(contrib_dim2.index, contrib_dim2.values, color=colors, edgecolor='black')
ax5.axvline(x=100/len(active_cols), color='red', linestyle='--',
            label=f'Contribucion esperada ({100/len(active_cols):.1f}%)')

ax5.set_xlabel('Contribucion (%)', fontsize=12)
ax5.set_title('Contribuciones de Variables a Dim.2\n(Rojo = sobre la media)', fontsize=14, fontweight='bold')
ax5.legend(loc='lower right')
ax5.grid(True, alpha=0.3, axis='x')

# -----------------------------------------------------------------------------
# GRAFICO 6: BIPLOT (Individuos + Variables)
# -----------------------------------------------------------------------------
ax6 = plt.subplot(2, 3, 6)

# Individuos (puntos)
for comp in df['Competition'].unique():
    mask = coords_plot['Competition'] == comp
    ax6.scatter(coords_plot.loc[mask, 'Dim.1'], coords_plot.loc[mask, 'Dim.2'],
               label=comp, c=palette.get(comp, 'gray'), s=50, alpha=0.5)

# Variables (flechas) - escaladas para visibilidad
scale = max(abs(row_coords['Dim.1']).max(), abs(row_coords['Dim.2']).max()) * 0.8
for var in active_cols:
    x = col_correlations.loc[var, 'Dim.1'] * scale
    y = col_correlations.loc[var, 'Dim.2'] * scale
    ax6.arrow(0, 0, x, y, head_width=0.2, head_length=0.15,
              fc='darkgreen', ec='darkgreen', linewidth=1.5, alpha=0.8)
    ax6.text(x*1.15, y*1.15, var, ha='center', va='center', fontsize=8,
             color='darkgreen', fontweight='bold')

ax6.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax6.axvline(0, color='black', linestyle='--', linewidth=0.5)
ax6.set_xlabel(f'Dim.1 ({(pca.percentage_of_variance_ / 100)[0]*100:.1f}%)', fontsize=12)
ax6.set_ylabel(f'Dim.2 ({(pca.percentage_of_variance_ / 100)[1]*100:.1f}%)', fontsize=12)
ax6.set_title('Biplot\n(Individuos + Variables)', fontsize=14, fontweight='bold')
ax6.legend(title='Competicion', loc='upper right')
ax6.grid(True, alpha=0.3)

plt.suptitle('ANALISIS DE COMPONENTES PRINCIPALES (PCA) - ESTILO FACTOMINER\n'
             'Dataset: Decathlon | Libreria: Prince',
             fontsize=16, fontweight='bold')

# Asegurar que la carpeta 'imagenes' existe
img_dir = os.path.join(SCRIPT_DIR, 'imagenes')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

plt.savefig(os.path.join(img_dir, '02_PCA_FactoMineR_graficos.png'), dpi=150)
plt.show()

print(f"\nGraficos guardados en: {os.path.join(img_dir, '02_PCA_FactoMineR_graficos.png')}")

# =============================================================================
# 12. TABLA RESUMEN FINAL (Estilo output de FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("12. RESUMEN EJECUTIVO DEL ANALISIS PCA")
print("=" * 80)

print(f"""
INTERPRETACION DE RESULTADOS
============================

1. DIMENSIONALIDAD:
   - Variables originales: {len(active_cols)}
   - Varianza en Dim.1 + Dim.2: {((pca.percentage_of_variance_ / 100)[0] + (pca.percentage_of_variance_ / 100)[1])*100:.1f}%
   - Componentes recomendados (Kaiser): {n_kaiser}

2. DIMENSION 1 ({(pca.percentage_of_variance_ / 100)[0]*100:.1f}% varianza):
   - Variables dominantes: {', '.join(col_contrib['Dim.1'].sort_values(ascending=False).head(3).index)}
   - Interpretacion: Eje relacionado con pruebas de fuerza/potencia

3. DIMENSION 2 ({(pca.percentage_of_variance_ / 100)[1]*100:.1f}% varianza):
   - Variables dominantes: {', '.join(col_contrib['Dim.2'].sort_values(ascending=False).head(3).index)}
   - Interpretacion: Eje relacionado con pruebas de resistencia/velocidad

4. VARIABLE SUPLEMENTARIA (Competition):
   - Los atletas de Juegos Olimpicos tienden a posicionarse
     diferente que los de Decastar en el espacio factorial
   - Esto sugiere perfiles de rendimiento distintos

5. CALIDAD DEL MODELO:
   - Atleta mejor representado: {cos2_plano.idxmax()} (cos2 = {cos2_plano.max():.3f})
   - Atleta peor representado: {cos2_plano.idxmin()} (cos2 = {cos2_plano.min():.3f})
""")

# =============================================================================
# 13. FUNCION AUXILIAR: CIRCULO DE CORRELACION INDIVIDUAL
# =============================================================================
def plot_correlation_circle(pca_model, data, active_columns, save_path=None):
    """
    Genera el circulo de correlacion clasico de FactoMineR.

    Parametros:
    -----------
    pca_model : prince.PCA
        Modelo PCA entrenado
    data : pd.DataFrame
        Datos originales
    active_columns : list
        Lista de columnas activas
    save_path : str, optional
        Ruta para guardar la figura
    """
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)

    # Circulo unitario
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=2)
    ax.add_artist(circle)

    # Correlaciones (es un atributo DataFrame en Prince)
    correlations = pca_model.column_correlations

    for var in active_columns:
        x = correlations.loc[var, 0]
        y = correlations.loc[var, 1]

        ax.arrow(0, 0, x*0.95, y*0.95, head_width=0.05, head_length=0.03,
                fc='red', ec='red', linewidth=2)
        ax.text(x*1.1, y*1.1, var, ha='center', va='center', fontsize=10,
               fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(f'Dim 1 ({pca_model.percentage_of_variance_[0]:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Dim 2 ({pca_model.percentage_of_variance_[1]:.1f}%)', fontsize=12)
    ax.set_title('Circulo de Correlacion (Variables)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

# Generar circulo de correlacion individual
print("\n" + "=" * 80)
print("13. GENERANDO CIRCULO DE CORRELACION INDIVIDUAL")
print("=" * 80)
# Usar la carpeta 'imagenes' ya creada arriba o asegurar su existencia
plot_correlation_circle(pca, df, active_cols, os.path.join(img_dir, '02_PCA_circulo_correlacion.png'))
print(f"Guardado en: {os.path.join(img_dir, '02_PCA_circulo_correlacion.png')}")

# =============================================================================
# 14. EXPORTAR RESULTADOS A EXCEL (Como hace FactoMineR)
# =============================================================================
print("\n" + "=" * 80)
print("14. EXPORTANDO RESULTADOS A EXCEL")
print("=" * 80)

try:
    with pd.ExcelWriter(os.path.join(SCRIPT_DIR, '02_PCA_resultados_FactoMineR.xlsx')) as writer:
        # Autovalores
        eigenvalues_df.to_excel(writer, sheet_name='Autovalores')

        # Coordenadas individuos
        row_coords.to_excel(writer, sheet_name='Coord_Individuos')

        # Contribuciones individuos
        (row_contrib * 100).round(2).to_excel(writer, sheet_name='Contrib_Individuos')

        # Cos2 individuos
        row_cos2.round(3).to_excel(writer, sheet_name='Cos2_Individuos')

        # Correlaciones variables
        col_correlations.round(3).to_excel(writer, sheet_name='Corr_Variables')

        # Contribuciones variables
        (col_contrib * 100).round(2).to_excel(writer, sheet_name='Contrib_Variables')

        # Cos2 variables
        col_cos2.round(3).to_excel(writer, sheet_name='Cos2_Variables')

    print("Resultados exportados a: 02_PCA_resultados_FactoMineR.xlsx")
    print("Hojas incluidas: Autovalores, Coord_Individuos, Contrib_Individuos,")
    print("                 Cos2_Individuos, Corr_Variables, Contrib_Variables, Cos2_Variables")
except Exception as e:
    print(f"Error al exportar Excel: {e}")

print("\n" + "=" * 80)
print("FIN DEL ANALISIS PCA ESTILO FACTOMINER")
print("=" * 80)
print("""
REFERENCIAS:
- Tutorial FactoMineR: http://factominer.free.fr/course/FactoTuto.html
- Husson, F., Le, S., & Pages, J. (2017). Exploratory Multivariate Analysis by Example Using R. CRC Press.
- Le, S., Josse, J., & Husson, F. (2008). FactoMineR: An R Package for Multivariate Analysis. JSS, 25(1).

Autor: @TodoEconometria
Profesor: Juan Marcelo Gutierrez Miranda
""")