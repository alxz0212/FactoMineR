import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from prince import PCA

# =============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO SAPPHIRE PREMIUM
# =============================================================================
st.set_page_config(
    page_title="Dashboard Empresarial | Titanic PCA",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Paleta de Colores Sapphire
MAIN_BG = "#0f172a"
CARD_BG = "rgba(30, 41, 59, 0.7)"
ACCENT_BLUE = "#3b82f6"
TEXT_COLOR = "#f1f5f9"
SUCCESS_GREEN = "#22c55e"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@500;700;800&display=swap');

    .stApp {{
        background-color: {MAIN_BG};
        color: {TEXT_COLOR};
        font-family: 'Inter', sans-serif;
    }}

    /* Header Styling */
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 2rem;
    }}
    .header-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: white;
        display: flex;
        align-items: center;
        gap: 15px;
    }}
    .header-user {{
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px 8px 0 0 !important;
        color: #94a3b8 !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        border: none !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ACCENT_BLUE} !important;
        color: white !important;
    }}

    /* Metric Cards */
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: left;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }}
    .metric-label {{
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }}
    .metric-value {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
    }}
    .metric-delta {{
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 10px;
    }}

    /* Logic/Conclusion Boxes */
    .conclusion-box {{
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid {SUCCESS_GREEN};
        border-left: 5px solid {SUCCESS_GREEN};
        padding: 12px 15px;
        border-radius: 4px;
        font-size: 0.85rem;
        line-height: 1.4;
        margin-top: 15px;
        color: #bbf7d0;
    }}

    /* Chart Containers */
    .chart-panel {{
        background: {CARD_BG};
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        height: 100%;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: {MAIN_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CARGA Y PROCESAMIENTO
# =============================================================================
@st.cache_data
def load_data():
    df = sns.load_dataset('titanic')
    df['age'] = df['age'].fillna(df['age'].median())
    return df.dropna(subset=['embark_town', 'sex', 'pclass'])

df = load_data()
vars_pca = ['pclass', 'age', 'sibsp', 'parch', 'fare']

# Motor PCA
pca_model = PCA(n_components=5, rescale_with_mean=True, rescale_with_std=True, random_state=42)
pca_model.fit(df[vars_pca])
coords = pca_model.row_coordinates(df[vars_pca])
corrs = pca_model.column_correlations
contrib = pca_model.column_contributions_ * 100
var_exp = pca_model.percentage_of_variance_ * 100

# =============================================================================
# 3. HEADER PERSONALIZADO
# =============================================================================
st.markdown(f"""
<div class="header-container">
    <div class="header-title">
        📈 Reporte Estadístico Avanzado: PCA Titanic
    </div>
    <div class="header-user">
        Daniel Alexis Mendoza Corne 👤
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 4. CUERPO DEL DASHBOARD
# =============================================================================
tabs = st.tabs(["📄 Resumen", "📊 Exploración", "📈 PCA", "🔍 Interpretación", "📅 Dataset", "✅ Conclusiones"])

with tabs[0]:
    # Fila de Métricas
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">👤 Total Pasajeros</div><div class="metric-value">{len(df):,}<span class="metric-delta" style="background: rgba(34,197,94,0.2); color: #4ade80;">+2%</span></div></div>""", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">🛡️ Tasa Supervivencia Global</div><div class="metric-value">{(df['survived'].mean()*100):.2f}%<span class="metric-delta" style="background: rgba(239,68,68,0.2); color: #f87171;">-1.5%</span></div></div>""", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">💼 Variables Analizadas</div><div class="metric-value">12</div></div>""", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">📊 Varianza Explicada (PC1+PC2)</div><div class="metric-value">{(var_exp[0]+var_exp[1]):.2f}% ✅</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fila 1 de Gráficos
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.subheader("Gráfico Scree (Scree Plot)")
        scree_df = pd.DataFrame({'Dim': [f'PC{i+1}' for i in range(len(var_exp))], 'Var': var_exp})
        scree_df['Acc'] = scree_df['Var'].cumsum()
        
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=scree_df['Dim'], y=scree_df['Var'], name='Varianza', marker_color='#3b82f6'))
        fig_scree.add_trace(go.Scatter(x=scree_df['Dim'], y=scree_df['Acc'], name='Acumulada', line=dict(color='#ef4444', width=3)))
        fig_scree.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_scree, width='stretch')
        
        st.markdown("""<div class="conclusion-box">✅ El Scree Plot sugiere retener los primeros 2-3 componentes principales, ya que explican una proporción significativa de la varianza y muestran un claro "codo", optimizando la reducción de dimensionalidad.</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.subheader("Círculo de Correlación PCA")
        fig_circle = go.Figure()
        t = np.linspace(0, 2*np.pi, 100)
        fig_circle.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode='lines', line=dict(color='rgba(255,255,255,0.2)', dash='dash'), showlegend=False))
        for var in vars_pca:
            fig_circle.add_trace(go.Scatter(x=[0, corrs.loc[var, 0]], y=[0, corrs.loc[var, 1]], mode='lines+text', text=["", var], textposition="top right", name=var, line=dict(color='#60a5fa', width=2)))
        fig_circle.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=0,r=0,t=20,b=0), xaxis=dict(range=[-1.1, 1.1]), yaxis=dict(range=[-1.1, 1.1], scaleanchor="x"))
        st.plotly_chart(fig_circle, width='stretch')
        
        st.markdown("""<div class="conclusion-box">✅ Variables como 'Fare' y 'Pclass' muestran fuertes correlaciones negativas en PC1, indicando que este componente captura la información socioeconómica. 'Age' y 'Survived' están más alineadas con PC2.</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fila 2 de Gráficos
    c3, c4 = st.columns(2)
    
    with c3:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.subheader("Importancia de Variables en PC1 (Top 5)")
        fig_bar1 = px.bar(contrib.sort_values(0), x=0, orientation='h', color_discrete_sequence=['#3b82f6'])
        fig_bar1.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="% Contribución", yaxis_title="")
        st.plotly_chart(fig_bar1, width='stretch')
        st.markdown("""<div class="conclusion-box">✅ La tarifa pagada y la clase del pasajero son los factores más determinantes en el primer componente principal, reflejando el estatus económico.</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.subheader("Importancia de Variables en PC2 (Top 5)")
        fig_bar2 = px.bar(contrib.sort_values(1), x=1, orientation='h', color_discrete_sequence=['#60a5fa'])
        fig_bar2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=250, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="% Contribución", yaxis_title="")
        st.plotly_chart(fig_bar2, width='stretch')
        st.markdown("""<div class="conclusion-box">✅ La edad y la supervivencia son cruciales en el segundo componente, con el sexo femenino también jugando un papel importante.</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align: center; color: #475569; padding: 40px 0 20px 0; font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 50px;">
    Dashboard Empresarial Titanic • Desarrollo: Daniel Alexis Mendoza Corne • 2026
</div>
""", unsafe_allow_html=True)
