import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from prince import PCA

# =============================================================================
# 1. CONFIGURACIÓN Y ESTADO GLOBAL (MOTOR ASOCIATIVO)
# =============================================================================
st.set_page_config(
    page_title="Qlik Sense PCA | Titanic",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos Premium - Look & Feel Pro (Glassmorphism + Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Outfit:wght@300;400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f0f2f6 0%, #e6e9ef 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Títulos con tipografía moderna */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    .selection-bar {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 15px 25px;
        border-left: 5px solid #009845;
        margin-bottom: 25px;
        display: flex;
        gap: 15px;
        align-items: center;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .qlik-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(4px);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out;
    }
    
    .qlik-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08);
    }

    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #1e293b;
        font-size: 2.2rem;
        font-weight: 800;
    }

    .filter-tag {
        background: white;
        border: 1px solid #009845;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        color: #009845;
        font-weight: 700;
        box-shadow: 0 2px 4px rgba(0,152,69,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Carga de datos base
@st.cache_data
def load_base_data():
    df_raw = sns.load_dataset('titanic')
    cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'embark_town']
    df = df_raw[cols].copy()
    df['age'] = df['age'].fillna(df['age'].median())
    df = df.dropna()
    # Identificador único para el "motor asociativo"
    df['id'] = range(len(df))
    return df

base_df = load_base_data()

# Inicialización del estado de filtros
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'sex': base_df['sex'].unique().tolist(),
        'class': base_df['class'].unique().tolist(),
        'survived': [0, 1]
    }

def reset_filters():
    st.session_state.filters = {
        'sex': base_df['sex'].unique().tolist(),
        'class': base_df['class'].unique().tolist(),
        'survived': [0, 1]
    }
    st.rerun()

# =============================================================================
# 2. SELECTION BAR (ESTILO QLIK)
# =============================================================================
st.markdown('<div class="selection-bar">', unsafe_allow_html=True)
col_bar1, col_bar2 = st.columns([8, 1])

with col_bar1:
    active_filters = []
    if len(st.session_state.filters['sex']) < len(base_df['sex'].unique()):
        active_filters.append(f"Sexo: {', '.join(st.session_state.filters['sex'])}")
    if len(st.session_state.filters['class']) < len(base_df['class'].unique()):
        active_filters.append(f"Clase: {', '.join(st.session_state.filters['class'])}")
    if len(st.session_state.filters['survived']) < 2:
        vals = ["Sí" if x == 1 else "No" for x in st.session_state.filters['survived']]
        active_filters.append(f"Sobrevivió: {', '.join(vals)}")
    
    if not active_filters:
        st.markdown("🔍 *Sin filtros activos (Mostrando todos los datos)*")
    else:
        st.markdown(" ".join([f'<span class="filter-tag">{f}</span>' for f in active_filters]), unsafe_allow_html=True)

with col_bar2:
    if st.button("Borrar Todo", type="primary"):
        reset_filters()
st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 3. SIDEBAR Y CÁLCULO ASOCIATIVO
# =============================================================================
with st.sidebar:
    st.header("🔍 Filtros de Selección")
    
    st.session_state.filters['sex'] = st.multiselect(
        "Género", 
        options=base_df['sex'].unique(), 
        default=st.session_state.filters['sex']
    )
    
    st.session_state.filters['class'] = st.multiselect(
        "Clase", 
        options=base_df['class'].unique(), 
        default=st.session_state.filters['class']
    )
    
    st.session_state.filters['survived'] = st.multiselect(
        "¿Sobrevivió?", 
        options=[0, 1], 
        default=st.session_state.filters['survived'],
        format_func=lambda x: "Sí" if x == 1 else "No"
    )

# Filtrar dataframe
df_filtered = base_df[
    (base_df['sex'].isin(st.session_state.filters['sex'])) & 
    (base_df['class'].isin(st.session_state.filters['class'])) & 
    (base_df['survived'].isin(st.session_state.filters['survived']))
]

# =============================================================================
# 4. DASHBOARD - KPI CARDS
# =============================================================================
k1, k2, k3, k4 = st.columns(4)

def kpi_card(col, label, value, total):
    col.markdown(f"""
    <div class="qlik-card">
        <div class="metric-title">{label}</div>
        <div class="metric-value">{value} <span style="font-size: 0.9rem; color: #8c8c8c;">/ {total}</span></div>
    </div>
    """, unsafe_allow_html=True)

kpi_card(k1, "Pasajeros Seleccionados", len(df_filtered), len(base_df))
kpi_card(k2, "Tasa de Supervivencia", f"{(df_filtered['survived'].mean()*100):.1f}%", f"{(base_df['survived'].mean()*100):.1f}%")
kpi_card(k3, "Edad Media", f"{df_filtered['age'].mean():.1f}", f"{base_df['age'].mean():.1f}")
kpi_card(k4, "Tarifa Promedio", f"${df_filtered['fare'].mean():.1f}", f"${base_df['fare'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# 5. GHOST CHARTS (EL CORAZÓN DE QLIK)
# =============================================================================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="qlik-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #111111; margin-top: 0;">Distribución por Edad (Selección vs Total)</h3>', unsafe_allow_html=True)
    
    # Crear gráfico de densidad (Ghost Chart)
    fig_age = go.Figure()
    
    # Total (Atentado - El "Fantasma")
    fig_age.add_trace(go.Histogram(
        x=base_df['age'],
        name='Total',
        nbinsx=30,
        marker_color='#d9d9d9',
        opacity=0.5
    ))
    
    # Selección (Resaltado)
    fig_age.add_trace(go.Histogram(
        x=df_filtered['age'],
        name='Seleccionado',
        nbinsx=30,
        marker_color='#009845'
    ))
    
    fig_age.update_layout(
        barmode='overlay',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=300
    )
    st.plotly_chart(fig_age, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="qlik-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #111111; margin-top: 0;">Clase vs Supervivencia</h3>', unsafe_allow_html=True)
    
    # Comparativa de clases resumida
    class_summ = df_filtered.groupby('class', observed=True)['survived'].mean().reset_index()
    class_total = base_df.groupby('class', observed=True)['survived'].mean().reset_index()
    
    fig_class = go.Figure()
    fig_class.add_trace(go.Bar(
        x=class_total['class'], y=class_total['survived'],
        name='Total', marker_color='#d9d9d9', opacity=0.5
    ))
    fig_class.add_trace(go.Bar(
        x=class_summ['class'], y=class_summ['survived'],
        name='Selección', marker_color='#009845'
    ))
    
    fig_class.update_layout(
        barmode='overlay',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    st.plotly_chart(fig_class, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 6. PCA ENGINE (ESTADO CONSCIENTE)
# =============================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="qlik-card">', unsafe_allow_html=True)
st.markdown('<h3 style="color: #111111; margin-top: 0;">Análisis PCA: Mapa Factorial de Pasajeros</h3>', unsafe_allow_html=True)

# Realizar PCA sobre el dataset COMPLETO para mantener los ejes estables
vars_pca = ['pclass', 'age', 'sibsp', 'parch', 'fare']
pca_engine = PCA(n_components=2, rescale_with_mean=True, rescale_with_std=True, random_state=42)
pca_engine.fit(base_df[vars_pca])
full_coords = pca_engine.row_coordinates(base_df[vars_pca])
full_coords['id'] = base_df['id'].values

# Identificar puntos seleccionados
selected_ids = df_filtered['id'].values
full_coords['isSelected'] = full_coords['id'].isin(selected_ids)

# Gráfico PCA Qlik-Style
fig_pca = go.Figure()

# Pasajeros NO seleccionados (Fantasmas)
non_selected = full_coords[~full_coords['isSelected']]
fig_pca.add_trace(go.Scattergl(
    x=non_selected[0], y=non_selected[1],
    mode='markers',
    marker=dict(color='#f0f0f0', size=4),
    name='Otros',
    hoverinfo='none'
))

# Pasajeros seleccionados (Resaltados)
selected = full_coords[full_coords['isSelected']]
fig_pca.add_trace(go.Scattergl(
    x=selected[0], y=selected[1],
    mode='markers',
    marker=dict(
        color='#009845', 
        size=6,
        line=dict(width=0.5, color='white')
    ),
    name='Seleccionados',
    text=[f"Pasajero {i}" for i in selected['id']]
))

fig_pca.update_layout(
    plot_bgcolor='white',
    xaxis=dict(title="Dimensión 1", showgrid=True, gridcolor='#f0f0f0'),
    yaxis=dict(title="Dimensión 2", showgrid=True, gridcolor='#f0f0f0'),
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    template="plotly_white"
)
st.plotly_chart(fig_pca, width='stretch')
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Dashboard Asociativo inspirado en Qlik Sense | Desarrollado con Streamlit + Plotly")
