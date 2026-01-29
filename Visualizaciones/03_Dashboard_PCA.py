
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prince import PCA
import requests
import io

# =============================================================================
# CONFIGURACION DE LA PAGINA
# =============================================================================
st.set_page_config(
    page_title="Dashboard PCA | Análisis FactoMineR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para "Look & Feel" Premium
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A; /* Azul oscuro profesional */
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        font-weight: 400;
        color: #64748B;
        margin-top: -10px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        color: #64748B;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# =============================================================================
@st.cache_data
def load_data():
    """Carga el dataset Decathlon y lo prepara."""
    url = "https://raw.githubusercontent.com/fhusson/FactoMineR/master/inst/extdata/decathlon.csv"
    try:
        # Intentar cargar desde URL con timberout corto para no colgar
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), index_col=0)
    except Exception as e:
        df = generate_fallback_data() # Si falla la red
    return df

def generate_fallback_data():
    """Genera datos simulados COMPLETOS si falla la conexión."""
    np.random.seed(42)
    atletas = ['SEBRLE', 'CLAY', 'KARPOV', 'BERNARD', 'YURKOV', 'WARNERS', 
               'ZSIVOCZKY', 'McMULLEN', 'MARTINEAU', 'HERNU', 'BARRAS', 'NOOL']
    
    # Simular las 10 pruebas (variables activas)
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
        # Variables suplementarias
        'Rank': np.arange(1, len(atletas)+1),
        'Points': np.random.randint(7500, 9000, len(atletas)),
        'Competition': np.random.choice(['OlympicG', 'Decastar'], len(atletas))
    }
    return pd.DataFrame(data, index=atletas)

@st.cache_resource
def compute_pca(df, n_components=5):
    """Calcula el PCA usando Prince."""
    # Seleccionar SOLO columnas numericas para el PCA, y excluir explicitamente las suplementarias conocidas
    # Esto evita el error "could not convert string to float"
    
    # 1. Identificar todas las numericas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 2. Excluir suplementarias conocidas si existen
    sup_cols_known = ['Rank', 'Points', 'Competition']
    active_cols = [c for c in numeric_cols if c not in sup_cols_known]
    
    # Seguridad: si despues de filtrar quedan muy pocas, usar las primeras numericas
    if len(active_cols) < 2:
         active_cols = numeric_cols[:10]

    # Limitar n_components al minimo entre (filas, columnas)
    n_components = min(n_components, len(active_cols), len(df))

    pca = PCA(n_components=n_components, rescale_with_mean=True, rescale_with_std=True, engine='sklearn', random_state=42)
    pca.fit(df[active_cols])
    return pca, active_cols

# =============================================================================
# LAYOUT PRINCIPAL
# =============================================================================

# Header
st.markdown('<div class="main-header">Análisis de Componentes Principales (PCA)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Metodología FactoMineR | Dataset: Decathlon</div>', unsafe_allow_html=True)
st.markdown("---")

# Carga de datos
df = load_data()
active_cols = df.columns[:10].tolist()
sup_cols_quali = ['Competition']

# Sidebar - Configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    n_comps = st.slider("Número de Dimensiones", 2, len(active_cols), 5)
    
    st.subheader("Ejes a Visualizar")
    x_axis = st.selectbox("Eje X", [f"DIM {i+1}" for i in range(n_comps)], index=0)
    y_axis = st.selectbox("Eje Y", [f"DIM {i+1}" for i in range(n_comps)], index=1)
    
    # Extraer indices de las dimensiones seleccionadas (0 para Dim 1, etc)
    dim_x = int(x_axis.split(" ")[1]) - 1
    dim_y = int(y_axis.split(" ")[1]) - 1

    st.info("""
    **Nota sobre la herramienta:**
    Este dashboard emula la salida de **FactoMineR** en R, proporcionando estadísticas clave para la interpretación exploratoria.
    """)

# Calculo PCA
pca, cols_usadas = compute_pca(df, n_comps)

# Preparar datos para gráficos
row_coords = pca.row_coordinates(df[active_cols])
col_coords = pca.column_correlations
eig = pca.eigenvalues_
var_exp = pca.percentage_of_variance_

# =============================================================================
# TABS DE CONTENIDO
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen & Inercia", "🎯 Variables (Círculo)", "🏃 Individuos (Mapa)", "📈 Biplot"])

# --- TAB 1: RESUMEN ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Estadísticas Globales")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Individuos (Atletas)</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-value">{len(active_cols)}</div>
            <div class="metric-label">Variables Activas</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-value">{sum(eig > 1)}</div>
            <div class="metric-label">Dim. Recomendadas (Kaiser)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Scree Plot (Varianza por Dimensión)")
        scree_df = pd.DataFrame({
            'Dimension': [f'Dim {i+1}' for i in range(len(eig))],
            'Varianza Explicada (%)': var_exp * 100,
            'Varianza Acumulada (%)': np.cumsum(var_exp) * 100,
            'Autovalor': eig
        })
        
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=scree_df['Dimension'], y=scree_df['Varianza Explicada (%)'],
            name='Varianza Indiv.', marker_color='#3B82F6'
        ))
        fig_scree.add_trace(go.Scatter(
            x=scree_df['Dimension'], y=scree_df['Varianza Acumulada (%)'],
            name='Acumulada', mode='lines+markers', line=dict(color='#EF4444', width=3)
        ))
        fig_scree.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_scree, width='stretch')
    
    st.markdown("### Tabla de Autovalores")
    st.dataframe(scree_df.set_index('Dimension').style.format("{:.2f}"), width='stretch')

# --- TAB 2: VARIABLES (CIRCULO DE CORRELACION) ---
with tab2:
    col_viz, col_data = st.columns([2, 1])
    
    with col_viz:
        st.subheader(f"Círculo de Correlación ({x_axis} vs {y_axis})")
        
        # Datos del circulo
        circle_data = col_coords.iloc[:, [dim_x, dim_y]].copy()
        circle_data.columns = ['x', 'y']
        circle_data['Variable'] = circle_data.index
        # Calcular Cos2 para colorear (Calidad de rep)
        # Nota: column_cosine_similarities_ a veces no tiene nombres de indice correctos en versions viejas, reaseguramos
        col_cos2 = pca.column_cosine_similarities_
        circle_data['Cos2'] = col_cos2.iloc[:, dim_x] + col_cos2.iloc[:, dim_y]
        
        fig_circle = go.Figure()
        
        # Circulo unidad
        fig_circle.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, line_color="gray", line_dash="dash")
        
        # Flechas
        for i, row in circle_data.iterrows():
            fig_circle.add_annotation(
                x=row['x'], y=row['y'], ax=0, ay=0,
                text=row['Variable'], xanchor="center", yanchor="bottom",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                arrowcolor=px.colors.sample_colorscale("Viridis", row['Cos2'])[0] # Color por calidad
            )
            
        fig_circle.update_xaxes(range=[-1.2, 1.2], zeroline=True, zerolinewidth=2, zerolinecolor='black', title=f"{x_axis} ({var_exp[dim_x]*100:.1f}%)")
        fig_circle.update_yaxes(range=[-1.2, 1.2], zeroline=True, zerolinewidth=2, zerolinecolor='black', title=f"{y_axis} ({var_exp[dim_y]*100:.1f}%)")
        fig_circle.update_layout(width=700, height=700, showlegend=False)
        
        st.plotly_chart(fig_circle, width='stretch')
        
    with col_data:
        st.subheader("Contribuciones a los Ejes")
        
        # Contribuciones
        contrib = pca.column_contributions_
        
        st.markdown(f"**Top Variables para {x_axis}**")
        top_x = contrib.iloc[:, dim_x].sort_values(ascending=False).head(5) * 100
        st.bar_chart(top_x, color="#6366F1")
        
        st.markdown(f"**Top Variables para {y_axis}**")
        top_y = contrib.iloc[:, dim_y].sort_values(ascending=False).head(5) * 100
        st.bar_chart(top_y, color="#EC4899")

# --- TAB 3: INDIVIDUOS (MAPA) ---
with tab3:
    st.subheader(f"Mapa de Individuos ({x_axis} vs {y_axis})")
    
    ind_data = row_coords.iloc[:, [dim_x, dim_y]].copy()
    ind_data.columns = ['x', 'y']
    ind_data['Individuo'] = ind_data.index
    ind_data['Competition'] = df['Competition'].values # Variable suplementaria
    
    fig_ind = px.scatter(
        ind_data, x='x', y='y', 
        color='Competition', 
        text='Individuo',
        hover_data=['Individuo'],
        title="Posición de los Atletas en el Espacio Factorial",
        color_discrete_map={'Decastar': '#3B82F6', 'OlympicG': '#EF4444'}
    )
    
    fig_ind.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig_ind.update_layout(
        xaxis_title=f"{x_axis} ({var_exp[dim_x]*100:.1f}%)",
        yaxis_title=f"{y_axis} ({var_exp[dim_y]*100:.1f}%)",
        height=700
    )
    fig_ind.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_ind.add_vline(x=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_ind, width='stretch')

# --- TAB 4: BIPLOT ---
with tab4:
    st.subheader("Biplot: Individuos + Variables")
    
    # Reutilizamos fig_ind como base
    fig_bi = go.Figure(fig_ind)
    
    # Factor de escalado para que las flechas se vean bien sobre los puntos
    scale_factor = max(ind_data['x'].abs().max(), ind_data['y'].abs().max()) * 0.8
    
    for i, row in circle_data.iterrows():
        fig_bi.add_shape(
            type='line', x0=0, y0=0, x1=row['x']*scale_factor, y1=row['y']*scale_factor,
            line=dict(color='green', width=1.5), opacity=0.5
        )
        fig_bi.add_annotation(
            x=row['x']*scale_factor, y=row['y']*scale_factor,
            text=row['Variable'],
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(color="green", size=10)
        )
        
    st.plotly_chart(fig_bi, width='stretch')


# Footer
st.markdown("---")
st.caption("Creado con Streamlit & Prince | Reproduciendo análisis de FactoMineR")
