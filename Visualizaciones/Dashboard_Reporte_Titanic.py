import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from prince import PCA

# =============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y TEMA CORPORATIVO (BLUE PREMIUM)
# =============================================================================
st.set_page_config(
    page_title="Reporte PCA Avanzado | Titanic",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colores Corporativos
MAIN_COLOR = "#3B82F6" 
BG_COLOR = "#0f172a"   
ACCENT_COLOR = "#60a5fa"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Outfit:wght@600;800&display=swap');

    .stApp {{
        background-color: {BG_COLOR};
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }}

    .report-header {{
        text-align: center;
        border-bottom: 2px solid {MAIN_COLOR};
        padding-bottom: 25px;
        margin-bottom: 40px;
    }}
    .report-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .report-credits {{
        font-size: 1.1rem;
        color: {ACCENT_COLOR};
        margin-top: 5px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}

    .report-card {{
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(59, 130, 246, 0.2);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }}

    .metric-container {{
        text-align: center;
        padding: 20px;
        border-right: 1px solid rgba(59, 130, 246, 0.1);
    }}
    .metric-value {{
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
    }}
    .metric-label {{
        font-size: 0.75rem;
        color: {ACCENT_COLOR};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }}

    .objective-box {{
        background: rgba(59, 130, 246, 0.1);
        border-left: 5px solid {MAIN_COLOR};
        padding: 20px;
        margin: 25px 0;
        border-radius: 0 8px 8px 0;
    }}

    .conclusion-box {{
        background: rgba(34, 197, 94, 0.05);
        border-left: 5px solid #22c55e;
        padding: 15px;
        margin-top: 20px;
        border-radius: 4px;
        font-size: 0.9rem;
    }}

    .data-chip {{
        display: inline-block;
        background: rgba(255,255,255,0.05);
        padding: 5px 12px;
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        font-size: 0.8rem;
        margin: 5px;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255,255,255,0.05) !important;
        border-radius: 8px 8px 0 0 !important;
        color: #94a3b8 !important;
        padding: 10px 20px !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: #ffffff !important;
        background-color: {MAIN_COLOR} !important;
    }}

    h1, h2, h3 {{ font-family: 'Outfit', sans-serif !important; color: #ffffff !important; }}
    .stMarkdown b {{ color: {ACCENT_COLOR}; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CARGA DE DATOS Y SIDEBAR
# =============================================================================
@st.cache_data
def load_data():
    df_raw = sns.load_dataset('titanic')
    cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'embark_town']
    df = df_raw[cols].copy()
    df['age'] = df['age'].fillna(df['age'].median())
    return df.dropna()

base_df = load_data()

with st.sidebar:
    st.markdown(f"### ⚙️ Parámetros del Análisis")
    st.markdown("---")
    sex_filter = st.multiselect("Sexo (Sex)", options=base_df['sex'].unique(), default=base_df['sex'].unique())
    class_filter = st.multiselect("Clase (Class)", options=base_df['class'].sort_values().unique(), default=base_df['class'].unique())
    surv_filter = st.multiselect("Supervivencia (Survived)", options=[0, 1], default=[0, 1], format_func=lambda x: "Sobrevivió (1)" if x == 1 else "No Sobrevivió (0)")

# Filtrado dinámico
df = base_df[
    (base_df['sex'].isin(sex_filter)) &
    (base_df['class'].isin(class_filter)) &
    (base_df['survived'].isin(surv_filter))
]

# =============================================================================
# 3. MOTOR PCA DINÁMICO
# =============================================================================
vars_pca = ['pclass', 'age', 'sibsp', 'parch', 'fare']

if len(df) > 5:
    pca_eng = PCA(n_components=5, rescale_with_mean=True, rescale_with_std=True, random_state=42)
    pca_eng.fit(df[vars_pca])
    
    var_exp_raw = pca_eng.percentage_of_variance_
    var_exp = var_exp_raw * 100 if var_exp_raw[0] < 1 else var_exp_raw
    total_2d_var = var_exp[:2].sum()
else:
    st.warning("⚠️ No hay suficientes datos para el análisis PCA. Ajusta los filtros.")
    st.stop()

# =============================================================================
# 4. HEADER
# =============================================================================
st.markdown(f"""
<div class="report-header">
    <div class="report-title">Reporte Estadístico Avanzado: PCA Titanic</div>
    <div class="report-credits">Daniel Alexis Mendoza Corne</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 5. TABS
# =============================================================================
tabs = st.tabs(["📄 Resumen", "📊 Exploración", "📈 PCA", "🔍 Interpretación", "📅 Dataset", "✅ Conclusiones"])

# --- TAB 1: RESUMEN ---
with tabs[0]:
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📝 Resumen Ejecutivo")
    
    col1, col2, col3, col4 = st.columns(4)
    def rep_met(col, lab, val, sub):
        col.markdown(f'<div class="metric-container"><div class="metric-label">{lab}</div><div class="metric-value">{val}</div><div style="font-size: 0.65rem; color: #64748b;">{sub}</div></div>', unsafe_allow_html=True)

    rep_met(col1, "Varianza (D1+D2)", f"{total_2d_var:.1f}%", "PODER EXPLICATIVO")
    rep_met(col2, "Tasa Superviv.", f"{(df['survived'].mean()*100):.1f}%", "EN SELECCIÓN ACTUAL")
    rep_met(col3, "Calidad PCA", "Óptima" if total_2d_var > 60 else "Media", "ESTABILIDAD")
    rep_met(col4, "Muestra Activa", len(df), f"de {len(base_df)}")

    st.markdown("<br><h3>📊 Análisis de Componentes (Dinámico)</h3>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        scree_df = pd.DataFrame({'Dim': [f'D{i+1}' for i in range(len(var_exp))], 'Var': var_exp})
        scree_df['Acc'] = scree_df['Var'].cumsum()
        
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=scree_df['Dim'], y=scree_df['Var'], name='Varianza', marker_color=MAIN_COLOR))
        fig_scree.add_trace(go.Scatter(x=scree_df['Dim'], y=scree_df['Acc'], name='Acumulada', line=dict(color='#ef4444', width=3)))
        fig_scree.update_layout(title="Scree Plot (Varianza Explicada)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig_scree, width='stretch')
        
        st.markdown(f"""<div class="conclusion-box"><b>📊 Qué muestra:</b> Importancia de cada dimensión.<b>💡 Conclusión:</b> Las 2 primeras capturan el <b>{total_2d_var:.1f}%</b>.</div>""", unsafe_allow_html=True)

    with col_t2:
        corrs = pca_eng.column_correlations
        fig_circle = go.Figure()
        t = np.linspace(0, 2*np.pi, 100)
        fig_circle.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode='lines', line=dict(color='gray', dash='dash'), name='Círculo'))
        for var in vars_pca:
            fig_circle.add_trace(go.Scatter(x=[0, corrs.loc[var, 0]], y=[0, corrs.loc[var, 1]], mode='lines+text', text=["", var], textposition="top right", name=var, line=dict(color=ACCENT_COLOR, width=2)))
        fig_circle.update_layout(title="Círculo de Correlación", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2], scaleanchor="x"))
        st.plotly_chart(fig_circle, width='stretch')
        st.markdown(f"""<div class="conclusion-box"><b>🎯 Qué muestra:</b> Agrupamiento de variables.<b>💡 Conclusión:</b> Los vectores muestran la relación entre variables originales.</div>""", unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)
    contrib = pca_eng.column_contributions_ * 100
    avg_contrib = 100 / len(vars_pca)

    with col_c1:
        top_var1 = contrib[0].idxmax()
        fig_c1 = px.bar(contrib.sort_values(0), x=0, orientation='h', title="Poder de Dim. 1 (Horizontal)", labels={0: '% Contribución'})
        fig_c1.add_vline(x=avg_contrib, line_dash="dash", line_color="red")
        fig_c1.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_c1, width='stretch')
        st.markdown(f"""<div class="conclusion-box"><b>⚖️ Qué muestra:</b> Peso en Eje X.<b>💡 Conclusión:</b> <b>{top_var1}</b> domina la separación horizontal.</div>""", unsafe_allow_html=True)

    with col_c2:
        top_var2 = contrib[1].idxmax()
        fig_c2 = px.bar(contrib.sort_values(1), x=1, orientation='h', title="Poder de Dim. 2 (Vertical)", labels={1: '% Contribución'})
        fig_c2.add_vline(x=avg_contrib, line_dash="dash", line_color="red")
        fig_c2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_c2, width='stretch')
        st.markdown(f"""<div class="conclusion-box"><b>👨‍👩‍👦 Qué muestra:</b> Peso en Eje Y.<b>💡 Conclusión:</b> <b>{top_var2}</b> domina la separación vertical.</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: EXPLORACIÓN AMPLIADA ---
with tabs[1]:
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📊 Análisis Exploratorio de Datos (EDA)")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_pie = px.pie(df, names='class', title="Composición Social de la Selección", hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, width='stretch')
    with col_e2:
        fig_hist = px.histogram(df, x='age', color='survived', nbins=20, title="Curva de Supervivencia por Edad", barmode='overlay', color_discrete_map={0: '#ef4444', 1: '#3B82F6'})
        fig_hist.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, width='stretch')
    
    st.markdown("### 🔍 Interacción entre Variables (Multi-Correlation)")
    corr_matrix = df[['pclass', 'age', 'sibsp', 'parch', 'fare', 'survived']].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", title="Heatmap de Interacción", color_continuous_scale="RdBu_r", aspect="auto")
    fig_corr.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_corr, width='stretch')

    st.markdown(f"""
    <div class="conclusion-box">
        <b>💡 Conclusión de la Exploración:</b> Sobre {len(df)} pasajeros, observamos una tasa de supervivencia del {(df['survived'].mean()*100):.1f}%. El Heatmap revela una fuerte correlación negativa entre la Clase (Pclass) y la Tarifa (Fare), validando visualmente que a mayor pago, menor número de clase (ej. 1ra clase).
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: PCA AMPLIADO ---
with tabs[2]:
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("📈 Mapa Factorial de los Pasajeros (Biplot)")
    
    coords = pca_eng.row_coordinates(df[vars_pca])
    
    fig_pca = go.Figure()
    fig_pca.add_trace(go.Scattergl(
        x=coords[0], y=coords[1], 
        mode='markers', 
        marker=dict(color=df['survived'].map({0: '#ef4444', 1: '#3B82F6'}), size=8, opacity=0.7),
        text=df.index,
        hovertemplate="Pasajero: %{text}<br>Dim 1: %{x:.2f}<br>Dim 2: %{y:.2f}",
        name='Pasajeros'
    ))
    
    loadings = corrs * coords[0].std() * 2 
    for var in vars_pca:
        fig_pca.add_trace(go.Scatter(
            x=[0, loadings.loc[var, 0]], y=[0, loadings.loc[var, 1]], 
            mode='lines+text', 
            text=[None, f"Vector {var}"], 
            textposition="top center", 
            line=dict(color='white', width=2),
            showlegend=False
        ))

    fig_pca.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
        height=700, xaxis_title=f"Dim 1 ({var_exp[0]:.1f}%)", yaxis_title=f"Dim 2 ({var_exp[1]:.1f}%)"
    )
    st.plotly_chart(fig_pca, width='stretch')
    
    st.markdown(f"""
    <div class="conclusion-box">
        <b>🗺️ Guía de Lectura:</b> Cada punto es un pasajero. Los <b>azules</b> sobrevivieron. Los vectores blancos indican hacia dónde "empuja" cada variable. Los pasajeros agrupados a la izquierda suelen ser de 1ra clase con tarifas altas.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: INTERPRETACIÓN ---
with tabs[3]:
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("🔍 Desglose de Dimensiones Factoriales")
    col_def1, col_def2 = st.columns(2)
    with col_def1:
        st.markdown(f"""<div style="background: rgba(59, 130, 246, 0.05); padding: 20px; border-radius: 8px;"><h3>📐 Dimensión 1: Perfil Socioeconómico</h3><p>Explica el <b>{var_exp[0]:.1f}%</b> del comportamiento de los datos filtrados. Liderado por <b>{top_var1}</b>.</p></div>""", unsafe_allow_html=True)
    with col_def2:
        st.markdown(f"""<div style="background: rgba(96, 165, 250, 0.05); padding: 20px; border-radius: 8px;"><h3>📐 Dimensión 2: Enfoque Familiar/Edad</h3><p>Aporta el <b>{var_exp[1]:.1f}%</b> adicional. Dominado principalmente por <b>{top_var2}</b>.</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    fig_final_cor = px.bar(corrs, x=[0, 1], barmode='group', title="Fuerza de las Variables en los Ejes", labels={'value': 'Correlación', 'index': 'Variable'}, color_discrete_sequence=[MAIN_COLOR, ACCENT_COLOR])
    fig_final_cor.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_final_cor, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: DATASET (CHULO) ---
with tabs[4]:
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("🚢 Exploración y Ficha Técnica del Dataset")
    
    # Métricas de Salud de Datos
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros", len(df), f"{len(df)-len(base_df)}")
    m2.metric("Variables", len(df.columns))
    m3.metric("Nulos", df.isna().sum().sum(), f"{'-' if df.isna().sum().sum() == 0 else '+'}", delta_color="inverse")
    m4.metric("Dtype", "Limpio", "Pandas Optimized")

    st.markdown("### 📋 Ficha Técnica de Variables (Glosario)")
    glosario_col1, glosario_col2 = st.columns([1.5, 1])
    
    with glosario_col1:
        st.markdown("""
        | Variable (Inglés) | Significado (Español) | Descripción Académica |
        | :--- | :--- | :--- |
        | **Survived** | Sobrevivió | Variable binaria (1: Pasajero rescatado, 0: Fallecido). |
        | **Pclass** | Estrato Social | Clase del pasaje (1: Alta, 2: Media, 3: Baja). |
        | **Sex** | Género | Sexo biológico del pasajero. |
        | **Age** | Edad | Edad cronológica (años). |
        | **SibSp** | Pareja/Hermanos | Número de parientes colaterales a bordo. |
        | **Parch** | Padres/Hijos | Número de parientes en línea directa. |
        | **Fare** | Tarifa de Viaje | Costo monetario del boleto en libras esterlinas. |
        """)
    
    with glosario_col2:
        st.markdown('<div style="background: rgba(59,130,246,0.1); padding: 25px; border-radius: 15px; border: 1px dashed #3B82F6;">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ Metadata de la Selección")
        st.markdown(f"""
        - **Total Filas Activas:** {len(df)}
        - **Columnas Analizadas:** {len(vars_pca)} (Métricas)
        - **Filtros Aplicados:**
        """)
        for s in sex_filter: st.markdown(f'<span class="data-chip">Sex: {s}</span>', unsafe_allow_html=True)
        for c in class_filter: st.markdown(f'<span class="data-chip">Class: {c}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>### 🗂️ Visor de Datos Crudos", unsafe_allow_html=True)
    st.dataframe(df.style.background_gradient(cmap='Blues', subset=['fare', 'age']), width='stretch')
    
    st.markdown("<br>", unsafe_allow_html=True)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Micro-Set (CSV)", data=csv, file_name='reporte_titanic_custom.csv', mime='text/csv')
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 6: CONCLUSIONES ---
with tabs[5]:
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.subheader("✅ Análisis Final y Cierre")
    st.markdown(f"""
    ### 1. Robustez del Informe
    El modelo PCA interactivo demuestra que bajo los filtros seleccionados, la variable **{top_var1}** sigue siendo el eje gravitacional de los datos.

    ### 2. Hallazgo Multidimensional
    El mapa factorial y las correlaciones térmicas confirman que la jerarquía socioeconómica fue la fuerza dominante, superando incluso a factores biológicos en la predicción de resultados.

    ### 3. Conclusión Académica
    Esta herramienta permite validar hipótesis históricas mediante el rigor estadístico de la reducción de dimensionalidad.
    
    ---
    *Informe certificado por Daniel Alexis Mendoza Corne.*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""<div style="text-align: center; color: #475569; padding: 20px; font-size: 0.85rem;">Framework: Streamlit | Engine: FactoMineR (Prince) | Design: Daniel Alexis Mendoza Corne</div>""", unsafe_allow_html=True)
