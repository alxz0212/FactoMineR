"""
-------------------------
Analisis Titanic: PCA con Estilo FactoMineR
-------------------------
Este script realiza un Analisis de Componentes Principales (PCA) sobre el dataset Titanic,
siguiendo la metodologia exploratoria de la libreria FactoMineR de R, pero implementado
en Python utilizando la libreria Prince.

Variables Activas (Cuantitativas):
- pclass: Clase del pasajero (1, 2, 3)
- age: Edad del pasajero (se imputan nulos con la mediana)
- sibsp: Numero de hermanos/conyuges a bordo
- parch: Numero de padres/hijos a bordo
- fare: Tarifa pagada

Variables Suplementarias (Cualitativas):
- survived: Si el pasajero sobrevivio (0 = No, 1 = Si)
- sex: Genero del pasajero
- class: Clase (Primera, Segunda, Tercera) - Redundante con pclass pero util para etiquetas
- embark_town: Ciudad de embarque
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince import PCA
import warnings
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

console = Console()

# Configuración inicial
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8-whitegrid')

# Obtener la ruta del directorio actual para guardar graficos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else '.'

console.rule("[bold cyan]PCA DEL DATASET TITANIC: ESTILO FACTOMINER[/bold cyan]")

# =============================================================================
# 1. CARGAR Y PREPROCESAR EL DATASET
# =============================================================================
console.print("\n[bold yellow]1. CARGANDO Y PREPROCESANDO EL DATASET[/bold yellow]")
titanic = sns.load_dataset('titanic')

# Seleccionar solo las columnas de interes para evitar ruido
cols_interes = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'embark_town']
df = titanic[cols_interes].copy()

# Manejo de valores nulos
null_table = Table(title="Valores nulos antes del procesamiento", style="red")
null_table.add_column("Variable", style="cyan")
null_table.add_column("Nulos", justify="right")
for col, val in df.isnull().sum().items():
    null_table.add_row(col, str(val))
console.print(null_table)

# Imputamos la edad con la mediana para no perder filas
df['age'] = df['age'].fillna(df['age'].median())

# Eliminamos filas con otros nulos minimos (como embark_town)
df = df.dropna()

console.print(f"\n[green]Dimensiones finales tras limpieza:[/green] [bold]{df.shape[0]}[/bold] pasajeros x [bold]{df.shape[1]}[/bold] columnas")
console.print("\n[bold cyan]Primeras filas del dataset:[/bold cyan]")
rprint(df.head())

# =============================================================================
# 2. DEFINICION DE VARIABLES
# =============================================================================
# Variables ACTIVAS: Construyen el espacio factorial
active_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']

# Variables SUPLEMENTARIAS: Se proyectan para explicar los grupos
sup_quali = ['survived', 'sex', 'class', 'embark_town']

print(f"\nVariables Activas (PCA): {active_cols}")
print(f"Variables Cualitativas Suplementarias (Color/Etiqueta): {sup_quali}")

# =============================================================================
# 3. ENTRENAMIENTO DEL MODELO PCA (PRINCE)
# =============================================================================
# Configuracion similar a PCA() de R: centrar y escalar (Standard Scaling)
pca = PCA(
    n_components=5,
    rescale_with_mean=True,
    rescale_with_std=True,
    copy=True,
    engine='sklearn',
    random_state=42
)

# El modelo se entrena solo con las cuantitativas
pca = pca.fit(df[active_cols])

console.print("\n[bold green][OK] Modelo PCA entrenado exitosamente con Prince.[/bold green]")

# =============================================================================
# 4. RESULTADOS DE AUTOVALORES (VARIACION EXPLICADA)
# =============================================================================
console.print(Panel("4. AUTOVALORES Y VARIANZA EXPLICADA", style="bold magenta"))

eig_table = Table(title="Resumen de Autovalores (pca$eig)", header_style="bold magenta")
eig_table.add_column("Componente", style="cyan")
eig_table.add_column("Autovalor", justify="right")
eig_table.add_column("% Varianza", justify="right")
eig_table.add_column("% Acumulado", justify="right")

summary_eig = pca.eigenvalues_summary
for i, row in summary_eig.iterrows():
    eig_table.add_row(
        str(i), 
        f"{float(row['eigenvalue']):.3f}", 
        str(row['% of variance']), 
        str(row['% of variance (cumulative)'])
    )
console.print(eig_table)

# =============================================================================
# 5. RESULTADOS DE LAS VARIABLES (CORRELACIONES Y CONTRIBUCIONES)
# =============================================================================
console.print(Panel("5. ANALISIS DE LAS VARIABLES", style="bold blue"))

# Correlaciones con las dimensiones
col_correlations = pca.column_correlations
corr_table = Table(title="Correlación de variables con los ejes", header_style="bold blue")
corr_table.add_column("Variable", style="cyan")
for i in range(5): corr_table.add_column(f"Dim.{i+1}", justify="right")

for var, rows in col_correlations.iterrows():
    corr_table.add_row(var, *[f"{v:.3f}" for v in rows])
console.print(corr_table)

# Contribuciones (%)
col_contrib = pca.column_contributions_ * 100
contrib_table = Table(title="Contribución (%) a la construcción de los ejes", header_style="bold green")
contrib_table.add_column("Variable", style="cyan")
for i in range(5): contrib_table.add_column(f"Dim.{i+1}", justify="right")

for var, rows in col_contrib.iterrows():
    contrib_table.add_row(var, *[f"{v:.2f}%" for v in rows])
console.print(contrib_table)

# =============================================================================
# 6. GENERACION DE GRAFICOS MULTIPLES
# =============================================================================
print("\nGenerando graficos en un dashboard...")

# Crear figura con mejor espacio para presentación
fig = plt.figure(figsize=(16, 10))

# --- 6.1 SCREE PLOT ---
ax1 = plt.subplot(2, 2, 1)
x_labels = [f'Dim.{i+1}' for i in range(len(pca.eigenvalues_))]
ax1.bar(x_labels, pca.percentage_of_variance_, color='#5dade2', edgecolor='black', alpha=0.8)
ax1.plot(x_labels, pca.percentage_of_variance_, 'ro-', linewidth=2, markersize=6, label='Individual')
ax1.plot(x_labels, pca.cumulative_percentage_of_variance_, 'go--', alpha=0.5, label='Acumulada')
ax1.set_title('Scree Plot: Varianza Explicada', fontsize=12, fontweight='bold', pad=15)
ax1.set_ylabel('% Varianza', fontsize=10)
ax1.set_ylim(0, 105)
ax1.legend(fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# --- 6.2 CIRCULO DE CORRELACION ---
ax2 = plt.subplot(2, 2, 2)
circle = plt.Circle((0, 0), 1, color='#ABB2B9', fill=False, linestyle='--', linewidth=1.5)
ax2.add_artist(circle)

# Dibujar vectores y etiquetas con mejor espaciado
for var in active_cols:
    x = col_correlations.loc[var, 0]
    y = col_correlations.loc[var, 1]
    
    # Vector
    ax2.arrow(0, 0, x*0.85, y*0.85, head_width=0.04, head_length=0.04, fc='#c0392b', ec='#c0392b', linewidth=1.5)
    
    # Ajuste de etiquetas para evitar colisiones (especialmente sibsp y parch)
    ha = 'left' if x > 0 else 'right'
    va = 'bottom' if y > 0 else 'top'
    # Offset dinámico
    offset_x = 1.15 if ha == 'left' else 1.15
    offset_y = 1.15 if va == 'bottom' else 1.15
    
    # Excepción para sibsp/parch que suelen estar muy juntos
    if var == 'sibsp': y += 0.05
    if var == 'parch': y -= 0.05

    ax2.text(x*offset_x, y*offset_y, var, color='#922b21', fontweight='bold', 
             fontsize=10, ha=ha, va=va)

ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax2.axvline(0, color='black', linewidth=0.8, alpha=0.5)
ax2.set_xlim(-1.3, 1.3)
ax2.set_ylim(-1.3, 1.3)
ax2.set_aspect('equal')
ax2.set_xlabel(f'Dim 1 ({pca.percentage_of_variance_[0]:.1f}%)', fontsize=10)
ax2.set_ylabel(f'Dim 2 ({pca.percentage_of_variance_[1]:.1f}%)', fontsize=10)
ax2.set_title('Círculo de Correlación (Variables)', fontsize=12, fontweight='bold', pad=15)
ax2.grid(linestyle=':', alpha=0.6)

# --- 6.3 MAPA DE INDIVIDUOS (SURVIVED) ---
ax3 = plt.subplot(2, 2, 3)
row_coords = pca.row_coordinates(df[active_cols])

# Colorear por si sobrevivio o no
sns.scatterplot(
    x=row_coords[0], y=row_coords[1], 
    hue=df['survived'], 
    palette={0: 'black', 1: 'green'}, 
    alpha=0.6, s=40, ax=ax3
)
ax3.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax3.axvline(0, color='black', linestyle='--', linewidth=0.5)
ax3.set_title('Mapa de Pasajeros (Coloreado por Supervivencia)', fontsize=14, fontweight='bold')
ax3.set_xlabel(f'Dim 1 ({pca.percentage_of_variance_[0]:.1f}%)')
ax3.set_ylabel(f'Dim 2 ({pca.percentage_of_variance_[1]:.1f}%)')
ax3.legend(title='Sobrevivió (1=Si)', loc='best')

# --- 6.4 CONTRIBUCIONES A DIM 1 ---
ax4 = plt.subplot(2, 2, 4)
contrib_dim1 = col_contrib[0].sort_values()
colors_contrib = ['#85c1e9' if v < 100/len(active_cols) else '#e74c3c' for v in contrib_dim1.values]
ax4.barh(contrib_dim1.index, contrib_dim1.values, color=colors_contrib, edgecolor='black', alpha=0.8)
ax4.axvline(x=100/len(active_cols), color='#c0392b', linestyle='--', label='Media Esperada')
ax4.set_title('Contribución de Variables a la Dimensión 1', fontsize=12, fontweight='bold', pad=15)
ax4.set_xlabel('% Contribución', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(axis='x', linestyle='--', alpha=0.5)

# --- PANEL INFORMATIVO (RESEÑA DE LAS DIMENSIONES) ---
resena_texto = (
    "RESEÑA DE DIMENSIONES:\n"
    "• Dim 1: Nivel Socioeconómico (Clase y Tarifa)\n"
    "• Dim 2: Estructura Familiar (Hijos, Padres, Hermanos)\n"
    "• Dim 3: Perfil de Edad (Variables de ciclo de vida)\n"
    "• Dim 4: Tamaño del Grupo Familiar (Detalle SibSp/Parch)\n"
    "• Dim 5: Variabilidad Residual de Estatus"
)
fig.text(0.5, 0.02, resena_texto, ha='center', fontsize=9, 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ABB2B9', boxstyle='round,pad=1'),
         fontfamily='sans-serif', linespacing=1.5)

# Ajuste global de la figura para que nada se corte
plt.suptitle('Análisis PCA: Dataset Titanic (Estilo FactoMineR)', fontsize=16, fontweight='bold', y=0.96)
plt.tight_layout(rect=[0.03, 0.12, 0.97, 0.93]) # Aumentamos margen inferior para la reseña
# Asegurar que la carpeta 'imagenes' existe
img_dir = os.path.join(SCRIPT_DIR, 'imagenes')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

plt.savefig(os.path.join(img_dir, 'Titanic_PCA_Resultados.png'), dpi=150, bbox_inches='tight')
# Maximizar la ventana del gráfico (específico para Windows)
try:
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
except AttributeError:
    # Si no es Windows o el backend no lo soporta, intentamos pantalla completa
    try:
        plt.get_current_fig_manager().full_screen_toggle()
    except:
        pass

plt.show()

# =============================================================================
# 7. INTERPRETACION FINAL Y RESEÑA DE DIMENSIONES
# =============================================================================
console.print(Panel.fit("7. RESEÑA DE LAS DIMENSIONES (Interpretación)", style="bold yellow"))

rprint(f"""
[bold cyan]DIMENSIÓN 1 ({pca.percentage_of_variance_[0]:.1f}%): NIVEL SOCIOECONÓMICO[/bold cyan]
- Dominada por [bold]pclass[/bold] (clase) y [bold]fare[/bold] (tarifa). 
- Separa a los pasajeros de élite (alta tarifa, 1ra clase) de los de recursos limitados.

[bold cyan]DIMENSIÓN 2 ({pca.percentage_of_variance_[1]:.1f}%): ESTRUCTURA FAMILIAR[/bold cyan]
- Dominada por [bold]sibsp[/bold] (hermanos/esposos) y [bold]parch[/bold] (padres/hijos). 
- Identifica si el pasajero viajaba solo o en un núcleo familiar.

[bold cyan]DIMENSIÓN 3 ({pca.percentage_of_variance_[2]:.1f}%): PERFIL DE EDAD[/bold cyan]
- La variable [bold]age[/bold] es el motor aquí.
- Diferencia a niños y jóvenes de adultos mayores dentro del barco.

[bold cyan]DIMENSIÓN 4 ({pca.percentage_of_variance_[3]:.1f}%): COMPOSICIÓN DEL GRUPO[/bold cyan]
- Refina la relación entre tipos de familiares. Ayuda a distinguir 
  matrimonios sin hijos de familias grandes.

[bold cyan]DIMENSIÓN 5 ({pca.percentage_of_variance_[4]:.1f}%): VARIABILIDAD RESIDUAL[/bold cyan]
- Captura matices finales entre clase y costo de boleto que no entraron en Dim 1.
""")

from rich.markup import escape
console.print(f"\n[bold green][OK][/bold green] Gráfico con reseña guardado en: [link file:///{escape(os.path.abspath(os.path.join(SCRIPT_DIR, 'Titanic_PCA_Resultados.png')))}]{escape(os.path.join(SCRIPT_DIR, 'Titanic_PCA_Resultados.png'))}[/link]")
console.print("[bold cyan]Analisis Finalizado.[/bold cyan]")
