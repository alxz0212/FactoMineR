import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from prince import PCA

# Configuración de la página
st.set_page_config(
    page_title="Titanic PCA Dashboard",
    page_icon="🚢",
    layout="wide"
)

# Estilo personalizado para corregir visibilidad en temas oscuros/claros
st.markdown("""
<style>
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    [data-testid="stMetricValue"] {
        color: #2e86c1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Título y Descripción
st.title("🚢 Análisis PCA Interactivo: Dataset Titanic")
st.markdown("""
Esta aplicación permite explorar el **Análisis de Componentes Principales (PCA)** del dataset Titanic.
Utilizamos la metodología **FactoMineR** para visualizar cómo se agrupan los pasajeros según su perfil.
""")

# =============================================================================
# 1. GUÍA DE VARIABLES (NUEVA SECCIÓN EN ESPAÑOL)
# =============================================================================
with st.expander("📖 Guía de Variables Activas (Significado en Español)"):
    st.markdown("""
    | Variable | Significado | Descripción |
    | :--- | :--- | :--- |
    | **pclass** | Clase del Pasajero | 1 = Primera, 2 = Segunda, 3 = Tercera. Refleja estatus socioeconómico. |
    | **age** | Edad | Años del pasajero. Los nulos se imputaron con la mediana (28 años). |
    | **sibsp** | Hermanos / Esposos | Número de hermanos o cónyuges a bordo. |
    | **parch** | Padres / Hijos | Número de padres o hijos a bordo. |
    | **fare** | Tarifa | Costo del boleto. Muy relacionado con la clase del pasajero. |
    """)

# =============================================================================
# 2. CARGA DE DATOS Y PROCESAMIENTO
# =============================================================================
@st.cache_data
def load_data():
    df_raw = sns.load_dataset('titanic')
    cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'embark_town']
    df = df_raw[cols].copy()
    df['age'] = df['age'].fillna(df['age'].median())
    df = df.dropna()
    return df

df = load_data()

# =============================================================================
# 3. SIDEBAR - FILTROS Y CONFIGURACIÓN
# =============================================================================
st.sidebar.header("⚙️ Configuración")

# Filtros Globales
st.sidebar.subheader("Filtros de Datos")
sexo_filtro = st.sidebar.multiselect("Género", options=df['sex'].unique(), default=df['sex'].unique())
clase_filtro = st.sidebar.multiselect("Clase", options=df['class'].sort_values().unique(), default=df['class'].unique())
sobrevivio_filtro = st.sidebar.multiselect("¿Sobrevivió?", options=[0, 1], default=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

# Filtrar dataframe
df_filtered = df[
    (df['sex'].isin(sexo_filtro)) & 
    (df['class'].isin(clase_filtro)) & 
    (df['survived'].isin(sobrevivio_filtro))
]

# Variables PCA
st.sidebar.subheader("Variables Activas")
vars_pca = st.sidebar.multiselect(
    "Variables para el cálculo",
    options=['pclass', 'age', 'sibsp', 'parch', 'fare'],
    default=['pclass', 'age', 'sibsp', 'parch', 'fare']
)

# =============================================================================
# 4. CÁLCULO DEL PCA
# =============================================================================
if len(vars_pca) >= 2:
    pca = PCA(
        n_components=min(len(vars_pca), 5),
        rescale_with_mean=True,
        rescale_with_std=True,
        random_state=42
    )
    pca = pca.fit(df_filtered[vars_pca])
    
    # Obtener resultados
    coords = pca.row_coordinates(df_filtered[vars_pca])
    corrs = pca.column_correlations
    var_exp = pca.percentage_of_variance_
    
    # Métricas principales
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pasajeros", len(df_filtered))
    m2.metric("Varianza D1", f"{var_exp[0]:.1f}%")
    m3.metric("Varianza D2", f"{var_exp[1]:.1f}%")
    m4.metric("Varianza Acum.", f"{var_exp[0]+var_exp[1]:.1f}%")

    # TABS PARA VISUALIZACIÓN
    tabs = st.tabs(["💎 Biplot Interactivo", "⭕ Círculo de Correlación", "📊 Varianza"])

    with tabs[0]:
        st.subheader("Mapa Biplot: Pasajeros y Variables")
        st.markdown("*Este gráfico combina la posición de los pasajeros con la influencia de las variables.*")
        
        # Crear Biplot Combinado
        fig_biplot = px.scatter(
            coords, x=0, y=1,
            color=df_filtered['survived'].astype(str),
            color_discrete_map={'0': 'black', '1': '#2ecc71'},
            symbol=df_filtered['sex'],
            marginal_x="box", marginal_y="box",
            hover_data={
                'Dim 1': coords[0],
                'Dim 2': coords[1],
                'Edad': df_filtered['age'],
                'Tarifa': df_filtered['fare']
            },
            labels={'0': 'Dimensión 1', '1': 'Dimensión 2', 'color': 'Sobrevivió', 'symbol': 'Sexo'},
            opacity=0.7
        )

        # Añadir vectores (flechas de variables) escalados
        scaling_factor = max(coords[0].abs().max(), coords[1].abs().max()) * 0.8
        for var in vars_pca:
            fig_biplot.add_trace(go.Scatter(
                x=[0, corrs.loc[var, 0] * scaling_factor],
                y=[0, corrs.loc[var, 1] * scaling_factor],
                mode='lines+text',
                text=["", f"<b>{var}</b>"],
                textposition="top center",
                line=dict(color='red', width=2),
                name=f"Efecto {var}",
                showlegend=False
            ))

        fig_biplot.update_layout(height=700, template="plotly_dark")
        st.plotly_chart(fig_biplot, width='stretch')

    with tabs[1]:
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.subheader("Círculo de Correlación")
            fig_circle = go.Figure()
            t = np.linspace(0, 2*np.pi, 100)
            fig_circle.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode='lines', line=dict(color='#5D6D7E', dash='dash'), name='Referencia'))
            
            for var in vars_pca:
                fig_circle.add_trace(go.Scatter(
                    x=[0, corrs.loc[var, 0]], y=[0, corrs.loc[var, 1]],
                    mode='lines+markers+text',
                    text=["", var], textposition="top right",
                    name=var, line=dict(width=3, color='#E74C3C'),
                    marker=dict(size=10, symbol="arrow-bar-up", angleref="previous")
                ))
            
            fig_circle.update_layout(
                xaxis=dict(range=[-1.2, 1.2], zeroline=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(range=[-1.2, 1.2], zeroline=True, gridcolor='rgba(255,255,255,0.1)', scaleanchor="x"),
                height=600, template="plotly_dark"
            )
            st.plotly_chart(fig_circle, width='stretch')
        
        with col_c2:
            st.markdown("### Tabla de Correlaciones")
            st.dataframe(corrs[[0, 1]].round(3).rename(columns={0: 'Dim 1', 1: 'Dim 2'}))

    with tabs[2]:
        st.subheader("Varianza Explicada")
        comp_df = pd.DataFrame({'Comp': [f'D.{i+1}' for i in range(len(var_exp))], 'Var': var_exp})
        fig_v = px.bar(comp_df, x='Comp', y='Var', text_auto='.1f', title="Scree Plot", color='Var', color_continuous_scale='Blues')
        st.plotly_chart(fig_v, width='stretch')

    # GUÍA DE DIMENSIONES MEJORADA
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Resumen de Dimensiones")
    st.sidebar.info("""
    **Dim 1 (Estatus):** Separa pasajeros por riqueza (Tarifa alta vs Clase baja).
    
    **Dim 2 (Familia):** Identifica grupos familiares grandes.
    
    **Dim 3 (Edad):** Captura diferencias por ciclo de vida.
    """)

else:
    st.error("Por favor, selecciona al menos 2 variables activas en la barra lateral para realizar el PCA.")

# Pie de página
st.markdown("---")
st.caption("Desarrollado con Streamlit y Plotly | Estilo FactoMineR")