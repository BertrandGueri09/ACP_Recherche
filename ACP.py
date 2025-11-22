# -*- coding: utf-8 -*-
"""
Dashboard Streamlit - Analyse en Composantes Principales
Thème: Performance historique UEFA Champions League (1955-2023)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="ACP - UEFA Champions League",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'image Postimages
postimg_url = "https://i.postimg.cc/vTp3B3LV/IMG-6442.jpg"

# Style CSS personnalisé avec image de fond du trophée Champions League
st.markdown(f"""
    <style>
    /* Image de fond - Trophée Champions League */
    .stApp {{
        background: linear-gradient(rgba(10, 25, 47, 0.85), rgba(18, 35, 60, 0.88)),
                    url('{postimg_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Conteneur principal */
    .main {
        padding: 1.5rem 2rem;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
    }
    
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, rgba(0, 51, 153, 0.95), rgba(0, 102, 204, 0.95));
        padding: 35px 25px;
        border-radius: 15px;
        text-align: center;
        border: 3px solid rgba(255, 215, 0, 0.7);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
        margin-bottom: 35px;
    }
    
    /* Titres principaux */
    h1 {
        color: #ffffff !important;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.8);
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
        margin-bottom: 15px !important;
        margin-top: 0 !important;
    }
    
    /* Sous-titres h2 */
    h2 {
        color: #ffffff !important;
        padding: 20px 20px;
        background: linear-gradient(135deg, rgba(0, 51, 153, 0.90), rgba(0, 102, 204, 0.90));
        border-radius: 10px;
        margin-top: 40px !important;
        margin-bottom: 25px !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        border-left: 6px solid #ffd700;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Sous-titres h3 */
    h3 {
        color: #ffffff !important;
        background: rgba(42, 82, 152, 0.75);
        padding: 15px 18px;
        border-radius: 8px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        margin-top: 30px !important;
        margin-bottom: 20px !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        border-left: 4px solid #ffd700;
    }
    
    /* Métriques avec espacement */
    .stMetric {
        background: linear-gradient(135deg, rgba(0, 51, 153, 0.95), rgba(0, 102, 204, 0.95));
        padding: 20px;
        border-radius: 12px;
        border: 2px solid rgba(255, 215, 0, 0.5);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        margin: 10px 0;
    }
    
    .stMetric label {
        color: #e0e0e0 !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffd700 !important;
        font-size: 28px !important;
        font-weight: 900 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #90EE90 !important;
        font-weight: 700 !important;
    }
    
    /* Texte général */
    .stMarkdown, p, label, .stSelectbox label, .stMultiSelect label, .stSlider label, .stTextInput label {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    
    /* DataFrames et tableaux */
    .dataframe {
        background-color: rgba(255, 255, 255, 0.97) !important;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
        margin: 20px 0;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #003399, #0066cc) !important;
        color: white !important;
        font-weight: 800 !important;
        text-align: center !important;
        padding: 12px !important;
        font-size: 14px !important;
    }
    
    .dataframe td {
        color: #000000 !important;
        background-color: rgba(255, 255, 255, 0.97) !important;
        padding: 10px !important;
        font-size: 13px !important;
    }
    
    /* Sidebar avec espacement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(0, 20, 60, 0.97), rgba(0, 40, 100, 0.97));
        border-right: 4px solid rgba(255, 215, 0, 0.6);
        padding: 20px 15px !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
        margin-top: 25px !important;
        margin-bottom: 15px !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] .stButton {
        margin-top: 20px !important;
        margin-bottom: 10px !important;
    }
    
    /* Boutons avec espacement */
    .stButton > button {
        background: linear-gradient(135deg, #003399, #0066cc);
        color: white;
        border: 3px solid #ffd700;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: 800;
        font-size: 17px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
        margin: 15px 0;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0044cc, #0088ff);
        border-color: #ffed4e;
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
    }
    
    /* Onglets avec espacement amélioré */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0, 30, 80, 0.85);
        border-radius: 12px;
        padding: 12px;
        gap: 8px;
        margin-bottom: 30px;
        margin-top: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #e0e0e0 !important;
        background: rgba(42, 82, 152, 0.7);
        border-radius: 8px;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 15px 25px;
        margin: 0 5px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(60, 100, 180, 0.85);
        border-color: rgba(255, 215, 0, 0.5);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #003399, #0066cc) !important;
        border: 3px solid #ffd700 !important;
        color: #ffd700 !important;
        font-weight: 900 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Contenu des onglets avec espacement */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 25px;
    }
    
    /* Input fields */
    .stTextInput input, .stSelectbox select, .stMultiSelect select {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #000000 !important;
        border: 2px solid rgba(0, 51, 153, 0.6);
        border-radius: 8px;
        font-weight: 600 !important;
        padding: 10px !important;
        font-size: 14px !important;
    }
    
    /* Sliders */
    .stSlider {
        margin: 20px 0;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: rgba(255, 255, 255, 0.3);
    }
    
    /* Checkbox et radio avec espacement */
    .stCheckbox, .stRadio {
        margin: 15px 0;
    }
    
    .stCheckbox label, .stRadio label {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
        font-size: 15px !important;
    }
    
    /* Messages d'info/succès/erreur avec espacement */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.96) !important;
        border-radius: 10px;
        border-left: 6px solid #003399;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        padding: 20px !important;
        margin: 20px 0 !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .stSuccess {
        background-color: rgba(144, 238, 144, 0.97) !important;
        color: #000000 !important;
        border-left: 6px solid #008000 !important;
    }
    
    .stWarning {
        background-color: rgba(255, 215, 0, 0.97) !important;
        color: #000000 !important;
        border-left: 6px solid #ff8c00 !important;
    }
    
    .stError {
        background-color: rgba(255, 99, 71, 0.97) !important;
        color: #000000 !important;
        border-left: 6px solid #dc143c !important;
    }
    
    .stInfo {
        background-color: rgba(173, 216, 230, 0.97) !important;
        color: #000000 !important;
        border-left: 6px solid #0066cc !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(42, 82, 152, 0.8) !important;
        color: #ffffff !important;
        border-radius: 8px;
        font-weight: 700 !important;
        padding: 15px !important;
        margin: 15px 0 !important;
    }
    
    /* Séparateur avec espacement */
    hr {
        border: 0;
        height: 3px;
        background: linear-gradient(to right, transparent, rgba(255, 215, 0, 0.7), transparent);
        margin: 35px 0;
    }
    
    /* Colonnes avec espacement */
    .row-widget.stHorizontal {
        gap: 20px;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00cc66, #00ff80);
        color: #000000;
        border: 2px solid #008844;
        border-radius: 8px;
        padding: 12px 25px;
        font-weight: 800;
        font-size: 15px;
        margin: 10px 0;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #00ff80, #00ffaa);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
    }
    
    /* Footer avec espacement */
    .footer {
        background: linear-gradient(135deg, rgba(0, 30, 80, 0.95), rgba(0, 51, 153, 0.95));
        padding: 25px;
        border-radius: 12px;
        border-top: 4px solid #ffd700;
        margin-top: 50px;
        box-shadow: 0 -4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Plotly graphs */
    .js-plotly-plot {
        background-color: rgba(255, 255, 255, 0.97) !important;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
        margin: 25px 0;
        padding: 10px;
    }
    
    /* Caption avec espacement */
    .css-1kyxreq, .stCaptionContainer {
        color: #e8e8e8 !important;
        font-style: italic;
        font-weight: 500 !important;
        margin-top: 8px !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
    }
    
    /* File uploader */
    .stFileUploader {
        margin: 20px 0;
    }
    
    /* Multiselect */
    .stMultiSelect {
        margin: 20px 0;
    }
    
    /* Selectbox */
    .stSelectbox {
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

@st.cache_data
def charger_donnees(uploaded_file=None):
    """Chargement du dataset UEFA Champions League"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv('UCL_AllTime_Performance_Table.csv')
        except FileNotFoundError:
            st.error("Fichier 'UCL_AllTime_Performance_Table.csv' non trouvé. Veuillez télécharger le fichier.")
            return None
    return df

def traiter_donnees(df, min_matches=10):
    """Traitement et enrichissement du dataset"""
    df_clean = df.copy()
    
    if 'goals' in df_clean.columns:
        goals_split = df_clean['goals'].str.split(':', expand=True)
        df_clean['Buts_Marques'] = pd.to_numeric(goals_split[0], errors='coerce')
        df_clean['Buts_Encaisses'] = pd.to_numeric(goals_split[1], errors='coerce')
    
    df_clean['Taux_Victoire'] = (df_clean['W'] / df_clean['M.'] * 100).round(2)
    df_clean['Taux_Defaite'] = (df_clean['L'] / df_clean['M.'] * 100).round(2)
    df_clean['Taux_Match_Nul'] = (df_clean['D'] / df_clean['M.'] * 100).round(2)
    df_clean['Points_par_Match'] = (df_clean['Pt.'] / df_clean['M.']).round(2)
    df_clean['Efficacite_Offensive'] = (df_clean['Buts_Marques'] / df_clean['M.']).round(2)
    df_clean['Solidite_Defensive'] = (df_clean['Buts_Encaisses'] / df_clean['M.']).round(2)
    df_clean['Ratio_Victoire_Defaite'] = (df_clean['W'] / (df_clean['L'] + 1)).round(2)
    
    df_clean = df_clean.dropna(subset=['Team', 'M.', 'W', 'L'])
    df_clean = df_clean[df_clean['M.'] >= min_matches]
    
    return df_clean

@st.cache_data
def effectuer_acp(df, variables_selectionnees):
    """Effectue l'analyse en composantes principales"""
    X = df[variables_selectionnees].copy()
    teams_names = df['Team'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    n_vars = len(variables_selectionnees)
    correlations = np.zeros((n_vars, min(2, X_pca.shape[1])))
    
    for i in range(n_vars):
        for j in range(min(2, X_pca.shape[1])):
            correlations[i, j] = np.corrcoef(X_scaled[:, i], X_pca[:, j])[0, 1]
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    contributions = (loadings**2) * 100
    
    return {
        'pca': pca,
        'X_pca': X_pca,
        'X_scaled': X_scaled,
        'scaler': scaler,
        'teams_names': teams_names,
        'correlations': correlations,
        'contributions': contributions,
        'variables': variables_selectionnees
    }

# =============================================================================
# INTERFACE PRINCIPALE
# =============================================================================

def main():
    
    # En-tête principal
    st.title("Analyse en Composantes Principales (ACP)")
    st.markdown("### Performance historique UEFA Champions League (1955-2023)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuration")
    st.sidebar.markdown("---")
    
    # Upload de fichier
    st.sidebar.subheader("1. Chargement des données")
    uploaded_file = st.sidebar.file_uploader(
        "Télécharger le fichier CSV", 
        type=['csv'],
        help="Format attendu: UCL_AllTime_Performance_Table.csv"
    )
    
    # Chargement des données
    df_original = charger_donnees(uploaded_file)
    
    if df_original is None:
        st.warning("Veuillez télécharger un fichier CSV ou vérifier que 'UCL_AllTime_Performance_Table.csv' existe.")
        st.stop()
    
    # Paramètres de filtrage
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Filtrage des données")
    min_matches = st.sidebar.slider(
        "Nombre minimum de matchs",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Filtrer les équipes avec moins de matchs"
    )
    
    # Traitement des données
    df = traiter_donnees(df_original, min_matches)
    
    st.sidebar.success(f"Données chargées: {len(df)} équipes")
    
    # Sélection des variables
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Variables pour l'ACP")
    
    variables_disponibles = [
        'M.', 'W', 'D', 'L', 'Dif', 'Pt.',
        'Buts_Marques', 'Buts_Encaisses',
        'Taux_Victoire', 'Taux_Defaite', 'Taux_Match_Nul',
        'Points_par_Match', 'Efficacite_Offensive', 
        'Solidite_Defensive', 'Ratio_Victoire_Defaite'
    ]
    
    variables_disponibles = [var for var in variables_disponibles if var in df.columns]
    
    variables_par_defaut = ['W', 'L', 'Dif', 'Taux_Victoire', 
                            'Points_par_Match', 'Efficacite_Offensive', 
                            'Solidite_Defensive']
    variables_par_defaut = [var for var in variables_par_defaut if var in variables_disponibles]
    
    variables_selectionnees = st.sidebar.multiselect(
        "Sélectionner les variables",
        options=variables_disponibles,
        default=variables_par_defaut,
        help="Minimum 3 variables requises"
    )
    
    if len(variables_selectionnees) < 3:
        st.error("Veuillez sélectionner au moins 3 variables pour l'ACP.")
        st.stop()
    
    # Bouton d'analyse
    st.sidebar.markdown("---")
    if st.sidebar.button("Lancer l'analyse ACP", type="primary"):
        st.session_state['acp_effectuee'] = True
    
    # ONGLETS PRINCIPAUX
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Données", 
        "Analyse ACP", 
        "Cercle des Corrélations",
        "Projection des Équipes",
        "Contributions",
        "Classements"
    ])
    
    # ONGLET 1: DONNÉES
    with tab1:
        st.header("Exploration des Données")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nombre d'équipes", len(df))
        with col2:
            st.metric("Total de matchs", int(df['M.'].sum()))
        with col3:
            st.metric("Variables disponibles", len(variables_disponibles))
        with col4:
            st.metric("Variables sélectionnées", len(variables_selectionnees))
        
        st.markdown("---")
        st.subheader("Aperçu des données")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            nb_lignes = st.selectbox("Nombre de lignes", [10, 20, 50, 100], index=0)
        with col2:
            equipe_recherche = st.text_input("Rechercher une équipe", "")
        
        if equipe_recherche:
            df_affichage = df[df['Team'].str.contains(equipe_recherche, case=False, na=False)]
        else:
            df_affichage = df.head(nb_lignes)
        
        st.dataframe(df_affichage, use_container_width=True, height=400)
        
        st.markdown("---")
        st.subheader("Statistiques descriptives")
        
        colonnes_stats = st.multiselect(
            "Sélectionner les colonnes pour les statistiques",
            options=variables_selectionnees,
            default=variables_selectionnees[:5] if len(variables_selectionnees) >= 5 else variables_selectionnees
        )
        
        if colonnes_stats:
            st.dataframe(df[colonnes_stats].describe().round(2), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Distribution des variables")
        
        variable_dist = st.selectbox("Sélectionner une variable", variables_selectionnees)
        
        fig = px.histogram(
            df, 
            x=variable_dist,
            nbins=30,
            title=f"Distribution de {variable_dist}",
            labels={variable_dist: variable_dist, 'count': 'Fréquence'},
            color_discrete_sequence=['#0066cc']
        )
        fig.update_layout(
            height=450,
            plot_bgcolor='rgba(255, 255, 255, 0.97)',
            paper_bgcolor='rgba(255, 255, 255, 0.97)',
            font=dict(color='#000000', size=13, family='Arial'),
            title=dict(
                font=dict(size=18, color='#003399', family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',
                title_font=dict(size=14, color='#003399', family='Arial Black')
            ),
            yaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',
                title_font=dict(size=14, color='#003399', family='Arial Black')
            ),
            bargap=0.1
        )
        fig.update_traces(
            marker=dict(
                color='#0066cc',
                line=dict(color='#003399', width=1.5)
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ONGLET 2: ANALYSE ACP
    with tab2:
        st.header("Résultats de l'Analyse en Composantes Principales")
        
        if 'acp_effectuee' not in st.session_state:
            st.info("Cliquez sur 'Lancer l'analyse ACP' dans la barre latérale pour commencer.")
            st.stop()
        
        with st.spinner("Analyse en cours..."):
            resultats_acp = effectuer_acp(df, variables_selectionnees)
        
        pca = resultats_acp['pca']
        X_pca = resultats_acp['X_pca']
        
        st.subheader("Variance expliquée")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CP1", 
                f"{pca.explained_variance_ratio_[0]*100:.1f}%",
                help="Variance expliquée par la première composante"
            )
        with col2:
            st.metric(
                "CP2", 
                f"{pca.explained_variance_ratio_[1]*100:.1f}%",
                help="Variance expliquée par la deuxième composante"
            )
        with col3:
            st.metric(
                "Total (CP1+CP2)", 
                f"{(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.1f}%",
                help="Variance totale expliquée par les deux premiers axes"
            )
        with col4:
            n_kaiser = np.sum(pca.explained_variance_ > 1)
            st.metric(
                "Composantes (Kaiser)", 
                n_kaiser,
                help="Nombre de composantes avec valeur propre > 1"
            )
        
        st.markdown("---")
        st.subheader("Tableau des valeurs propres")
        
        valeurs_propres_df = pd.DataFrame({
            'Composante': [f'CP{i+1}' for i in range(len(pca.explained_variance_))],
            'Valeur propre': pca.explained_variance_,
            'Variance expliquée (%)': pca.explained_variance_ratio_ * 100,
            'Variance cumulée (%)': np.cumsum(pca.explained_variance_ratio_) * 100
        })
        
        st.dataframe(valeurs_propres_df.round(2), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Visualisation des valeurs propres")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Scatter(
                x=list(range(1, len(pca.explained_variance_) + 1)),
                y=pca.explained_variance_,
                mode='lines+markers',
                name='Valeurs propres',
                line=dict(color='#003399', width=3),
                marker=dict(size=10, color='#0066cc', line=dict(color='#003399', width=2))
            ))
            fig_scree.add_hline(
                y=1, 
                line_dash="dash", 
                line_color="red",
                line_width=2,
                annotation_text="Critère de Kaiser",
                annotation_position="right",
                annotation=dict(font=dict(size=12, color='red', family='Arial Black'))
            )
            fig_scree.update_layout(
                title=dict(
                    text="Scree Plot - Valeurs Propres",
                    font=dict(size=16, color='#003399', family='Arial Black')
                ),
                xaxis_title="Composante",
                yaxis_title="Valeur propre",
                height=450,
                plot_bgcolor='rgba(255, 255, 255, 0.97)',
                paper_bgcolor='rgba(255, 255, 255, 0.97)',
                font=dict(color='#000000', size=12, family='Arial'),
                xaxis=dict(
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    title_font=dict(size=13, color='#003399', family='Arial Black')
                ),
                yaxis=dict(
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    title_font=dict(size=13, color='#003399', family='Arial Black')
                )
            )
            st.plotly_chart(fig_scree, use_container_width=True)
        
        with col2:
            fig_var = go.Figure()
            fig_var.add_trace(go.Scatter(
                x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                y=np.cumsum(pca.explained_variance_ratio_) * 100,
                mode='lines+markers',
                name='Variance cumulée',
                line=dict(color='#00cc66', width=3),
                marker=dict(size=10, color='#00ff80', line=dict(color='#00cc66', width=2)),
                fill='tozeroy',
                fillcolor='rgba(0, 204, 102, 0.2)'
            ))
            fig_var.add_hline(
                y=80, 
                line_dash="dash", 
                line_color="red",
                line_width=2,
                annotation_text="80% de variance",
                annotation_position="right",
                annotation=dict(font=dict(size=12, color='red', family='Arial Black'))
            )
            fig_var.update_layout(
                title=dict(
                    text="Variance Expliquée Cumulée",
                    font=dict(size=16, color='#003399', family='Arial Black')
                ),
                xaxis_title="Composante",
                yaxis_title="Variance cumulée (%)",
                height=450,
                plot_bgcolor='rgba(255, 255, 255, 0.97)',
                paper_bgcolor='rgba(255, 255, 255, 0.97)',
                font=dict(color='#000000', size=12, family='Arial'),
                xaxis=dict(
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    title_font=dict(size=13, color='#003399', family='Arial Black')
                ),
                yaxis=dict(
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    title_font=dict(size=13, color='#003399', family='Arial Black')
                )
            )
            st.plotly_chart(fig_var, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Interprétation des axes principaux")
        
        correlations = resultats_acp['correlations']
        corr_df = pd.DataFrame({
            'Variable': variables_selectionnees,
            'CP1': correlations[:, 0],
            'CP2': correlations[:, 1] if correlations.shape[1] > 1 else 0
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**CP1 ({pca.explained_variance_ratio_[0]*100:.1f}% de variance)**")
            cp1_pos = corr_df[corr_df['CP1'] > 0.5].sort_values('CP1', ascending=False)
            if len(cp1_pos) > 0:
                st.write("Variables positivement corrélées:")
                for _, row in cp1_pos.iterrows():
                    st.write(f"- {row['Variable']} ({row['CP1']:.3f})")
            
            cp1_neg = corr_df[corr_df['CP1'] < -0.5].sort_values('CP1')
            if len(cp1_neg) > 0:
                st.write("Variables négativement corrélées:")
                for _, row in cp1_neg.iterrows():
                    st.write(f"- {row['Variable']} ({row['CP1']:.3f})")
        
        with col2:
            st.write(f"**CP2 ({pca.explained_variance_ratio_[1]*100:.1f}% de variance)**")
            cp2_pos = corr_df[corr_df['CP2'] > 0.5].sort_values('CP2', ascending=False)
            if len(cp2_pos) > 0:
                st.write("Variables positivement corrélées:")
                for _, row in cp2_pos.iterrows():
                    st.write(f"- {row['Variable']} ({row['CP2']:.3f})")
            
            cp2_neg = corr_df[corr_df['CP2'] < -0.5].sort_values('CP2')
            if len(cp2_neg) > 0:
                st.write("Variables négativement corrélées:")
                for _, row in cp2_neg.iterrows():
                    st.write(f"- {row['Variable']} ({row['CP2']:.3f})")
    
    # ONGLET 3: CERCLE DES CORRÉLATIONS
    with tab3:
        st.header("Cercle des Corrélations")
        
        if 'acp_effectuee' not in st.session_state:
            st.info("Effectuez d'abord l'analyse ACP.")
            st.stop()
        
        resultats_acp = effectuer_acp(df, variables_selectionnees)
        correlations = resultats_acp['correlations']
        pca = resultats_acp['pca']
        
        fig = go.Figure()
        
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_hline(y=0, line_color='gray', line_width=1)
        fig.add_vline(x=0, line_color='gray', line_width=1)
        
        colors = ['#003399', '#0066cc', '#ff6600', '#cc0000', '#00cc66', '#9933ff', '#ff33cc', '#ffcc00']
        
        for i, var in enumerate(variables_selectionnees):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=[0, correlations[i, 0]],
                y=[0, correlations[i, 1]],
                mode='lines+markers+text',
                name=var,
                line=dict(width=3, color=color),
                marker=dict(size=[0, 12], color=color),
                text=['', var],
                textposition='top center',
                textfont=dict(size=11, color='black', family='Arial Black'),
                hovertemplate=f'<b>{var}</b><br>CP1: %{{x:.3f}}<br>CP2: %{{y:.3f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=f"Cercle des Corrélations (CP1: {pca.explained_variance_ratio_[0]*100:.1f}% vs CP2: {pca.explained_variance_ratio_[1]*100:.1f}%)",
                font=dict(size=18, color='#003399', family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=dict(
                text=f"CP1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                font=dict(size=14, color='#003399', family='Arial Black')
            ),
            yaxis_title=dict(
                text=f"CP2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                font=dict(size=14, color='#003399', family='Arial Black')
            ),
            xaxis=dict(
                range=[-1.2, 1.2], 
                constrain='domain',
                gridcolor='rgba(200, 200, 200, 0.5)',
                zerolinecolor='rgba(100, 100, 100, 0.8)',
                zerolinewidth=2
            ),
            yaxis=dict(
                range=[-1.2, 1.2], 
                scaleanchor='x', 
                scaleratio=1,
                gridcolor='rgba(200, 200, 200, 0.5)',
                zerolinecolor='rgba(100, 100, 100, 0.8)',
                zerolinewidth=2
            ),
            height=750,
            showlegend=True,
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.97)',
                bordercolor='#003399',
                borderwidth=2,
                font=dict(size=11, color='#000000')
            ),
            plot_bgcolor='rgba(255, 255, 255, 0.97)',
            paper_bgcolor='rgba(255, 255, 255, 0.97)',
            font=dict(color='#000000', size=12, family='Arial')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Tableau des corrélations")
        
        corr_display = pd.DataFrame({
            'Variable': variables_selectionnees,
            'CP1': correlations[:, 0],
            'CP2': correlations[:, 1] if correlations.shape[1] > 1 else 0,
            'Distance à l\'origine': np.sqrt(correlations[:, 0]**2 + (correlations[:, 1] if correlations.shape[1] > 1 else 0)**2)
        })
        
        st.dataframe(
            corr_display.round(3).sort_values('Distance à l\'origine', ascending=False),
            use_container_width=True
        )
    
    # ONGLET 4: PROJECTION DES ÉQUIPES
    with tab4:
        st.header("Projection des Équipes sur le Plan Factoriel")
        
        if 'acp_effectuee' not in st.session_state:
            st.info("Effectuez d'abord l'analyse ACP.")
            st.stop()
        
        resultats_acp = effectuer_acp(df, variables_selectionnees)
        X_pca = resultats_acp['X_pca']
        teams_names = resultats_acp['teams_names']
        pca = resultats_acp['pca']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            afficher_labels = st.checkbox("Afficher les noms", value=True)
        with col2:
            nb_equipes_labels = st.slider("Nombre d'équipes étiquetées", 0, len(teams_names), 15)
        with col3:
            colorer_par = st.selectbox(
                "Colorer par",
                ["Aucun", "Quadrant", "Points", "Taux de victoire"]
            )
        
        projection_df = pd.DataFrame({
            'Equipe': teams_names,
            'CP1': X_pca[:, 0],
            'CP2': X_pca[:, 1],
            'Points': df['Pt.'].values,
            'Taux_Victoire': df['Taux_Victoire'].values,
            'Matchs': df['M.'].values
        })
        
        projection_df['Quadrant'] = 'Q4'
        projection_df.loc[(projection_df['CP1'] > 0) & (projection_df['CP2'] > 0), 'Quadrant'] = 'Q1'
        projection_df.loc[(projection_df['CP1'] < 0) & (projection_df['CP2'] > 0), 'Quadrant'] = 'Q2'
        projection_df.loc[(projection_df['CP1'] < 0) & (projection_df['CP2'] < 0), 'Quadrant'] = 'Q3'
        projection_df.loc[(projection_df['CP1'] > 0) & (projection_df['CP2'] < 0), 'Quadrant'] = 'Q4'
        
        projection_df['Distance'] = np.sqrt(projection_df['CP1']**2 + projection_df['CP2']**2)
        equipes_extremes = projection_df.nlargest(nb_equipes_labels, 'Distance')['Equipe'].tolist()
        
        if colorer_par == "Quadrant":
            color_col = 'Quadrant'
            color_scale = None
        elif colorer_par == "Points":
            color_col = 'Points'
            color_scale = 'Viridis'
        elif colorer_par == "Taux de victoire":
            color_col = 'Taux_Victoire'
            color_scale = 'RdYlGn'
        else:
            color_col = None
            color_scale = None
        
        fig = px.scatter(
            projection_df,
            x='CP1',
            y='CP2',
            color=color_col,
            color_continuous_scale=color_scale,
            hover_data=['Equipe', 'Points', 'Taux_Victoire', 'Matchs'],
            title=f"Projection des Équipes (CP1: {pca.explained_variance_ratio_[0]*100:.1f}% vs CP2: {pca.explained_variance_ratio_[1]*100:.1f}%)",
            labels={
                'CP1': f'CP1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'CP2': f'CP2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
            }
        )
        
        if afficher_labels:
            for _, row in projection_df[projection_df['Equipe'].isin(equipes_extremes)].iterrows():
                fig.add_annotation(
                    x=row['CP1'],
                    y=row['CP2'],
                    text=row['Equipe'],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor='gray',
                    font=dict(size=10, color='black', family='Arial Black'),
                    bgcolor='rgba(255, 255, 255, 0.97)',
                    bordercolor='#003399',
                    borderwidth=2,
                    borderpad=4
                )
        
        fig.add_hline(y=0, line_color='gray', line_width=0.5, line_dash='dash')
        fig.add_vline(x=0, line_color='gray', line_width=0.5, line_dash='dash')
        
        fig.update_layout(
            height=750,
            plot_bgcolor='rgba(255, 255, 255, 0.97)',
            paper_bgcolor='rgba(255, 255, 255, 0.97)',
            font=dict(color='#000000', size=12, family='Arial'),
            title=dict(
                font=dict(size=18, color='#003399', family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',
                zerolinecolor='rgba(100, 100, 100, 0.5)',
                title_font=dict(size=14, color='#003399', family='Arial Black')
            ),
            yaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',
                zerolinecolor='rgba(100, 100, 100, 0.5)',
                title_font=dict(size=14, color='#003399', family='Arial Black')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Analyse par quadrants")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            q1 = len(projection_df[projection_df['Quadrant'] == 'Q1'])
            st.metric("Quadrant 1 (++)", q1)
            st.caption("Performance élevée sur les deux axes")
        
        with col2:
            q2 = len(projection_df[projection_df['Quadrant'] == 'Q2'])
            st.metric("Quadrant 2 (-+)", q2)
            st.caption("CP1 faible, CP2 élevé")
        
        with col3:
            q3 = len(projection_df[projection_df['Quadrant'] == 'Q3'])
            st.metric("Quadrant 3 (--)", q3)
            st.caption("Performance faible sur les deux axes")
        
        with col4:
            q4 = len(projection_df[projection_df['Quadrant'] == 'Q4'])
            st.metric("Quadrant 4 (+-)", q4)
            st.caption("CP1 élevé, CP2 faible")
        
        st.markdown("---")
        st.subheader("Recherche d'équipe spécifique")
        equipe_recherche = st.selectbox("Sélectionner une équipe", sorted(teams_names))
        
        if equipe_recherche:
            equipe_data = projection_df[projection_df['Equipe'] == equipe_recherche].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CP1", f"{equipe_data['CP1']:.3f}")
            with col2:
                st.metric("CP2", f"{equipe_data['CP2']:.3f}")
            with col3:
                st.metric("Quadrant", equipe_data['Quadrant'])
            with col4:
                st.metric("Distance origine", f"{equipe_data['Distance']:.3f}")
    
    # ONGLET 5: CONTRIBUTIONS
    with tab5:
        st.header("Contributions des Variables")
        
        if 'acp_effectuee' not in st.session_state:
            st.info("Effectuez d'abord l'analyse ACP.")
            st.stop()
        
        resultats_acp = effectuer_acp(df, variables_selectionnees)
        contributions = resultats_acp['contributions']
        pca = resultats_acp['pca']
        
        st.subheader("Tableau des contributions")
        
        contrib_df = pd.DataFrame({
            'Variable': variables_selectionnees,
            'Contribution CP1 (%)': contributions[:, 0],
            'Contribution CP2 (%)': contributions[:, 1] if contributions.shape[1] > 1 else 0
        })
        
        st.dataframe(
            contrib_df.round(2).sort_values('Contribution CP1 (%)', ascending=False),
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("Visualisation des contributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cp1 = go.Figure()
            fig_cp1.add_trace(go.Bar(
                x=variables_selectionnees,
                y=contributions[:, 0],
                marker=dict(
                    color=contributions[:, 0],
                    colorscale='Blues',
                    showscale=False,
                    line=dict(color='#003399', width=2)
                ),
                text=contributions[:, 0].round(1),
                textposition='outside',
                texttemplate='%{text}%',
                textfont=dict(size=12, color='#000000', family='Arial Black'),
                hovertemplate='<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>'
            ))
            fig_cp1.update_layout(
                title=dict(
                    text=f"Contributions à CP1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                    font=dict(size=16, color='#003399', family='Arial Black')
                ),
                xaxis_title="Variables",
                yaxis_title="Contribution (%)",
                height=450,
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(255, 255, 255, 0.97)',
                paper_bgcolor='rgba(255, 255, 255, 0.97)',
                font=dict(color='#000000', size=11, family='Arial'),
                xaxis=dict(
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    title_font=dict(size=12, color='#003399', family='Arial Black')
                ),
                yaxis=dict(
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    title_font=dict(size=12, color='#003399', family='Arial Black')
                )
            )
            st.plotly_chart(fig_cp1, use_container_width=True)
        
        with col2:
            fig_cp2 = go.Figure()
            fig_cp2.add_trace(go.Bar(
                x=variables_selectionnees,
                y=contributions[:, 1] if contributions.shape[1] > 1 else np.zeros(len(variables_selectionnees)),
                marker=dict(
                    color=contributions[:, 1] if contributions.shape[1] > 1 else np.zeros(len(variables_selectionnees)),
                    colorscale='Oranges',
                    showscale=False,
                    line=dict(color='#ff6600', width=2)
                ),
                text=(contributions[:, 1] if contributions.shape[1] > 1 else np.zeros(len(variables_selectionnees))).round(1),
                textposition='outside',
                texttemplate='%{text}%',
                textfont=dict(size=12, color='#000000', family='Arial Black'),
                hovertemplate='<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>'
            ))
            fig_cp2.update_layout(
                title=dict(
                    text=f"Contributions à CP2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                    font=dict(size=16, color='#003399', family='Arial Black')
                ),
                xaxis_title="Variables",
                yaxis_title="Contribution (%)",
                height=450,
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(255, 255, 255, 0.97)',
                paper_bgcolor='rgba(255, 255, 255, 0.97)',
                font=dict(color='#000000', size=11, family='Arial'),
                xaxis=dict(
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    title_font=dict(size=12, color='#003399', family='Arial Black')
                ),
                yaxis=dict(
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    title_font=dict(size=12, color='#003399', family='Arial Black')
                )
            )
            st.plotly_chart(fig_cp2, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Contributions cumulées")
        
        contrib_df['Contribution Totale (%)'] = contrib_df['Contribution CP1 (%)'] + contrib_df['Contribution CP2 (%)']
        contrib_sorted = contrib_df.sort_values('Contribution Totale (%)', ascending=False)
        
        fig_cumul = go.Figure()
        fig_cumul.add_trace(go.Bar(
            x=contrib_sorted['Variable'],
            y=contrib_sorted['Contribution CP1 (%)'],
            name='CP1',
            marker=dict(color='#0066cc', line=dict(color='#003399', width=1)),
            hovertemplate='<b>%{x}</b><br>CP1: %{y:.2f}%<extra></extra>'
        ))
        fig_cumul.add_trace(go.Bar(
            x=contrib_sorted['Variable'],
            y=contrib_sorted['Contribution CP2 (%)'],
            name='CP2',
            marker=dict(color='#ff9933', line=dict(color='#ff6600', width=1)),
            hovertemplate='<b>%{x}</b><br>CP2: %{y:.2f}%<extra></extra>'
        ))
        
        fig_cumul.update_layout(
            title=dict(
                text="Contributions Cumulées (CP1 + CP2)",
                font=dict(size=16, color='#003399', family='Arial Black')
            ),
            xaxis_title="Variables",
            yaxis_title="Contribution (%)",
            barmode='stack',
            height=450,
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(255, 255, 255, 0.97)',
            paper_bgcolor='rgba(255, 255, 255, 0.97)',
            font=dict(color='#000000', size=12, family='Arial'),
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.97)',
                bordercolor='#003399',
                borderwidth=2,
                font=dict(size=11, color='#000000', family='Arial Black')
            ),
            xaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',
                title_font=dict(size=12, color='#003399', family='Arial Black')
            ),
            yaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',
                title_font=dict(size=12, color='#003399', family='Arial Black')
            )
        )
        
        st.plotly_chart(fig_cumul, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Analyse des contributions")
        
        var_importante_cp1 = contrib_df.nlargest(3, 'Contribution CP1 (%)')[['Variable', 'Contribution CP1 (%)']]
        var_importante_cp2 = contrib_df.nlargest(3, 'Contribution CP2 (%)')[['Variable', 'Contribution CP2 (%)']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 3 variables pour CP1:**")
            for _, row in var_importante_cp1.iterrows():
                st.write(f"- {row['Variable']}: {row['Contribution CP1 (%)']:.1f}%")
        
        with col2:
            st.write("**Top 3 variables pour CP2:**")
            for _, row in var_importante_cp2.iterrows():
                st.write(f"- {row['Variable']}: {row['Contribution CP2 (%)']:.1f}%")
    
    # ONGLET 6: CLASSEMENTS
    with tab6:
        st.header("Classements des Équipes")
        
        if 'acp_effectuee' not in st.session_state:
            st.info("Effectuez d'abord l'analyse ACP.")
            st.stop()
        
        resultats_acp = effectuer_acp(df, variables_selectionnees)
        X_pca = resultats_acp['X_pca']
        teams_names = resultats_acp['teams_names']
        
        classement_df = pd.DataFrame({
            'Equipe': teams_names,
            'CP1': X_pca[:, 0],
            'CP2': X_pca[:, 1],
            'Matchs': df['M.'].values,
            'Victoires': df['W'].values,
            'Points': df['Pt.'].values,
            'Dif': df['Dif'].values,
            'Taux_Victoire': df['Taux_Victoire'].values
        })
        
        classement_df['Rang_CP1'] = classement_df['CP1'].rank(ascending=False).astype(int)
        classement_df['Rang_CP2'] = classement_df['CP2'].rank(ascending=False).astype(int)
        
        st.markdown("---")
        type_classement = st.radio(
            "Sélectionner le type de classement",
            ["Par CP1 (Performance globale)", "Par CP2", "Par Points totaux", "Par Taux de victoire"],
            horizontal=True
        )
        
        nb_equipes = st.slider("Nombre d'équipes à afficher", 10, len(teams_names), 20, step=5)
        
        st.markdown("---")
        if type_classement == "Par CP1 (Performance globale)":
            classement_affiche = classement_df.nlargest(nb_equipes, 'CP1')[
                ['Rang_CP1', 'Equipe', 'CP1', 'CP2', 'Points', 'Victoires', 'Matchs', 'Taux_Victoire']
            ]
            titre_classement = f"Top {nb_equipes} - Classement par CP1"
        
        elif type_classement == "Par CP2":
            classement_affiche = classement_df.nlargest(nb_equipes, 'CP2')[
                ['Rang_CP2', 'Equipe', 'CP2', 'CP1', 'Points', 'Victoires', 'Matchs', 'Taux_Victoire']
            ]
            titre_classement = f"Top {nb_equipes} - Classement par CP2"
        
        elif type_classement == "Par Points totaux":
            classement_affiche = classement_df.nlargest(nb_equipes, 'Points')[
                ['Equipe', 'Points', 'CP1', 'CP2', 'Victoires', 'Matchs', 'Taux_Victoire']
            ]
            classement_affiche.insert(0, 'Rang', range(1, len(classement_affiche) + 1))
            titre_classement = f"Top {nb_equipes} - Classement par Points"
        
        else:
            classement_affiche = classement_df.nlargest(nb_equipes, 'Taux_Victoire')[
                ['Equipe', 'Taux_Victoire', 'CP1', 'CP2', 'Points', 'Victoires', 'Matchs']
            ]
            classement_affiche.insert(0, 'Rang', range(1, len(classement_affiche) + 1))
            titre_classement = f"Top {nb_equipes} - Classement par Taux de victoire"
        
        st.subheader(titre_classement)
        st.dataframe(
            classement_affiche.round(3).reset_index(drop=True),
            use_container_width=True,
            height=600
        )
        
        st.markdown("---")
        st.subheader("Comparaison graphique")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            equipes_comparaison = st.multiselect(
                "Sélectionner des équipes à comparer",
                sorted(teams_names),
                default=sorted(teams_names)[:5]
            )
        
        with col2:
            if equipes_comparaison:
                data_comparaison = classement_df[classement_df['Equipe'].isin(equipes_comparaison)]
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name='CP1',
                    x=data_comparaison['Equipe'],
                    y=data_comparaison['CP1'],
                    marker=dict(
                        color='#0066cc',
                        line=dict(color='#003399', width=2),
                        pattern=dict(shape="/")
                    ),
                    text=data_comparaison['CP1'].round(2),
                    textposition='outside',
                    textfont=dict(size=11, color='#000000', family='Arial Black'),
                    hovertemplate='<b>%{x}</b><br>CP1: %{y:.3f}<extra></extra>'
                ))
                fig_comp.add_trace(go.Bar(
                    name='CP2',
                    x=data_comparaison['Equipe'],
                    y=data_comparaison['CP2'],
                    marker=dict(
                        color='#ff9933',
                        line=dict(color='#ff6600', width=2),
                        pattern=dict(shape="\\")
                    ),
                    text=data_comparaison['CP2'].round(2),
                    textposition='outside',
                    textfont=dict(size=11, color='#000000', family='Arial Black'),
                    hovertemplate='<b>%{x}</b><br>CP2: %{y:.3f}<extra></extra>'
                ))
                
                fig_comp.update_layout(
                    title=dict(
                        text="Comparaison CP1 vs CP2",
                        font=dict(size=16, color='#003399', family='Arial Black')
                    ),
                    xaxis_title="Équipes",
                    yaxis_title="Score",
                    barmode='group',
                    height=450,
                    xaxis_tickangle=-45,
                    plot_bgcolor='rgba(255, 255, 255, 0.97)',
                    paper_bgcolor='rgba(255, 255, 255, 0.97)',
                    font=dict(color='#000000', size=12, family='Arial'),
                    legend=dict(
                        bgcolor='rgba(255, 255, 255, 0.97)',
                        bordercolor='#003399',
                        borderwidth=2,
                        font=dict(size=11, color='#000000', family='Arial Black')
                    ),
                    xaxis=dict(
                        gridcolor='rgba(200, 200, 200, 0.5)',
                        title_font=dict(size=12, color='#003399', family='Arial Black')
                    ),
                    yaxis=dict(
                        gridcolor='rgba(200, 200, 200, 0.5)',
                        title_font=dict(size=12, color='#003399', family='Arial Black')
                    )
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Statistiques du classement")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Équipe leader CP1", classement_df.nlargest(1, 'CP1')['Equipe'].values[0])
        with col2:
            st.metric("Équipe leader CP2", classement_df.nlargest(1, 'CP2')['Equipe'].values[0])
        with col3:
            st.metric("Plus de points", classement_df.nlargest(1, 'Points')['Equipe'].values[0])
        with col4:
            st.metric("Meilleur taux victoire", classement_df.nlargest(1, 'Taux_Victoire')['Equipe'].values[0])
        
        st.markdown("---")
        st.subheader("Export des résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_classement = classement_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger le classement complet (CSV)",
                data=csv_classement,
                file_name='classement_acp_ucl.csv',
                mime='text/csv'
            )
        
        with col2:
            csv_top = classement_affiche.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Télécharger le Top {nb_equipes} (CSV)",
                data=csv_top,
                file_name=f'top_{nb_equipes}_ucl.csv',
                mime='text/csv'
            )
    
    # FOOTER
    st.markdown("---")
    
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Analyse en Composantes Principales**")
        st.caption("UEFA Champions League 1955-2023")
    
    with col2:
        st.markdown("**Données**")
        st.caption(f"{len(df)} équipes analysées")
    
    with col3:
        st.markdown("**Variables**")
        st.caption(f"{len(variables_selectionnees)} variables sélectionnées")
    
    st.markdown('</div>', unsafe_allow_html=True)

# POINT D'ENTRÉE
if __name__ == "__main__":
    main()
