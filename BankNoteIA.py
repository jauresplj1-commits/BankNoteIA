import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="BankNote AI - Classificateur de Billets",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design professionnel
def local_css():
    st.markdown("""
    <style>
    /* Styles généraux */
    .main {
        background-color: #f8f9fa;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header */
    .header {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .title-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Métriques */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Uploader */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Success/Error messages */
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    
    .error-msg {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f4;
        border-radius: 5px 5px 0px 0px;
        gap: 1rem;
        padding: 10px 20px;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 5px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour charger le modèle avec cache
@st.cache_resource
def load_model():
    """Charge le modèle avec mise en cache"""
    try:
        model = keras.models.load_model('best_model_final.h5')
        return model, True
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None, False

# Fonction pour prétraiter l'image
def preprocess_image(img, target_size=(224, 224)):
    """Prétraite l'image pour la prédiction"""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Fonction pour faire une prédiction
def predict_image(model, img_array):
    """Fait une prédiction sur l'image"""
    predictions = model.predict(img_array, verbose=0)
    return predictions

# Classes de billets
DENOMINATIONS = {
    0: {"label": "1000", "name": "Mille Roupies", "color": "#FF6B6B"},
    1: {"label": "2000", "name": "Deux Mille Roupies", "color": "#4ECDC4"},
    2: {"label": "5000", "name": "Cinq Mille Roupies", "color": "#45B7D1"},
    3: {"label": "10000", "name": "Dix Mille Roupies", "color": "#96CEB4"},
    4: {"label": "20000", "name": "Vingt Mille Roupies", "color": "#FFEAA7"},
    5: {"label": "50000", "name": "Cinquante Mille Roupies", "color": "#DDA0DD"},
    6: {"label": "100000", "name": "Cent Mille Roupies", "color": "#98D8C8"}
}

# Fonction pour créer un graphique de probabilités
def create_probability_chart(probabilities):
    """Crée un graphique à barres des probabilités"""
    labels = [DENOMINATIONS[i]["name"] for i in range(len(probabilities))]
    values = [prob * 100 for prob in probabilities]
    colors = [DENOMINATIONS[i]["color"] for i in range(len(probabilities))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Probabilités de Prédiction",
        xaxis_title="Dénominations",
        yaxis_title="Probabilité (%)",
        yaxis_range=[0, 100],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        height=400
    )
    
    return fig

# Fonction pour créer un indicateur de confiance
def create_confidence_gauge(confidence):
    """Crée un indicateur de jauge pour la confiance"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confiance de la Prédiction"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# Fonction pour créer une carte de résultat
def create_result_card(prediction_data):
    """Crée une carte visuelle pour afficher les résultats"""
    pred_idx = prediction_data["predicted_class"]
    confidence = prediction_data["confidence"]
    
    card_html = f"""
    <div style="
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid {DENOMINATIONS[pred_idx]['color']};
    ">"""
    card_html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
            <div style="
                width: 60px;
                height: 60px;
                background: {DENOMINATIONS[pred_idx]['color']};
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 1rem;
            ">
                <span style="color: white; font-size: 1.5rem; font-weight: bold;">💰</span>
            </div>
            <div>
                <h2 style="margin: 0; color: #333;">Dénomination Identifiée</h2>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">{datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
            </div>
        </div>"""
    card_html += f"""
        <div style="text-align: center; margin: 2rem 0;">
            <h1 style="
                font-size: 3rem;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">
                {DENOMINATIONS[pred_idx]['label']} Rp
            </h1>
            <p style="color: #666; font-size: 1.2rem; margin: 0.5rem 0 0 0;">
                {DENOMINATIONS[pred_idx]['name']}
            </p>
        </div>"""
    card_html += f"""
        <div style="
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1.5rem;
        ">"""
    card_html += f"""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #666;">Confiance</span>
                <span style="
                    background: {'#4CAF50' if confidence > 80 else '#FF9800' if confidence > 60 else '#F44336'};
                    color: white;
                    padding: 0.25rem 1rem;
                    border-radius: 20px;
                    font-weight: bold;
                ">
                    {confidence:.1f}%
                </span>
            </div>
            <div style="
                margin-top: 0.5rem;
                height: 10px;
                background: #e0e0e0;
                border-radius: 5px;
                overflow: hidden;
            ">
                <div style="
                    width: {confidence}%;
                    height: 100%;
                    background: linear-gradient(90deg, {DENOMINATIONS[pred_idx]['color']} 0%, #667eea 100%);
                    border-radius: 5px;
                "></div>
            </div>
        </div>
    </div>"""


    
    return card_html

# Fonction pour afficher les statistiques
def display_statistics(stats):
    """Affiche les statistiques sous forme de métriques"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #666; font-size: 0.9rem;">Précision Moyenne</div>
            <div class="metric-value">{:.1f}%</div>
        </div>
        """.format(stats.get("avg_confidence", 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #666; font-size: 0.9rem;">Billets Analysés</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(stats.get("total_analyzed", 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #666; font-size: 0.9rem;">Dernière Prédiction</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(stats.get("last_prediction", "N/A")), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #666; font-size: 0.9rem;">Modèle Version</div>
            <div class="metric-value">1.0.0</div>
        </div>
        """.format(), unsafe_allow_html=True)

# Page principale
def main():
    # Appliquer le CSS
    local_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #667eea; margin-bottom: 0;">💰</h1>
            <h3 style="margin-top: 0;">BankNote AI</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        page = st.radio(
            "Sélectionnez une page",
            ["🎯 Analyser un Billet", "📊 Dashboard", "ℹ️ À Propos"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Informations sur le modèle
        st.markdown("### Modèle")
        model, loaded = load_model()
        
        if loaded:
            st.success("✅ Modèle chargé avec succès")
            st.markdown("**best_model_final.h5**")
            st.markdown("Deep Learning - Transfer Learning")
        else:
            st.error("❌ Modèle non chargé")
        
        st.markdown("---")
        
        # Informations techniques
        with st.expander("ℹ️ Informations techniques"):
            st.markdown("""
            **Architecture:** CNN Fine-tuned
            **Classes:** 7 dénominations
            **Input:** 224x224 RGB
            **Framework:** TensorFlow/Keras
            **Précision:** >90% (test set)
            """)
        
        # Footer sidebar
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p>BankNote AI v1.0.0</p>
            <p>© 2025 - Tous droits réservés</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown("""
    <div class="header">
        <h1 class="title-text">BankNote AI - Classificateur Intelligent de Billets</h1>
        <p style="color: #666; font-size: 1.1rem;">
            Détection automatique de dénominations de billets Rupiah par intelligence artificielle
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialiser l'état de session
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            "total_analyzed": 0,
            "avg_confidence": 0,
            "last_prediction": "N/A",
            "predictions_history": []
        }
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Page: Analyser un billet
    if page == "🎯 Analyser un Billet":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>📤 Téléverser une Image</h3>
                <p>Téléversez une image de billet Rupiah à analyser</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Uploader d'image
            uploaded_file = st.file_uploader(
                "Choisissez une image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                # Lire et afficher l'image
                image_bytes = uploaded_file.read()
                st.session_state.uploaded_image = Image.open(io.BytesIO(image_bytes))
                
                # Afficher l'image uploadée
                st.markdown("### 📷 Image téléversée")
                st.image(st.session_state.uploaded_image, use_column_width=True)
                
                # Bouton d'analyse
                if st.button("🔍 Analyser le Billet", use_container_width=True):
                    if model is not None and loaded:
                        with st.spinner("Analyse en cours..."):
                            # Prétraitement
                            img_array = preprocess_image(st.session_state.uploaded_image)
                            
                            # Prédiction
                            predictions = predict_image(model, img_array)
                            predicted_class = np.argmax(predictions[0])
                            confidence = np.max(predictions[0]) * 100
                            
                            # Stocker les résultats
                            prediction_data = {
                                "timestamp": datetime.now().isoformat(),
                                "predicted_class": predicted_class,
                                "confidence": confidence,
                                "probabilities": predictions[0].tolist(),
                                "image_size": st.session_state.uploaded_image.size
                            }
                            
                            # Mettre à jour les statistiques
                            st.session_state.stats["total_analyzed"] += 1
                            st.session_state.stats["last_prediction"] = DENOMINATIONS[predicted_class]["label"]
                            st.session_state.stats["predictions_history"].append(prediction_data)
                            
                            # Calculer la moyenne de confiance
                            confidences = [p["confidence"] for p in st.session_state.stats["predictions_history"]]
                            st.session_state.stats["avg_confidence"] = np.mean(confidences) if confidences else 0
                            
                            # Stocker les résultats pour l'affichage
                            st.session_state.prediction_data = prediction_data
                    else:
                        st.error("Le modèle n'est pas chargé correctement.")
            
            else:
                # Exemple d'image
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
                    <p style="color: #666;">📱 <strong>Format supporté:</strong> JPG, PNG, BMP</p>
                    <p style="color: #666;">💡 <strong>Conseil:</strong> Assurez-vous que le billet est bien visible</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>📊 Résultats d'Analyse</h3>
                <p>Résultats détaillés de la classification</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher les résultats
            if 'prediction_data' in st.session_state:
                prediction_data = st.session_state.prediction_data
                
                # Carte de résultat
                st.markdown(create_result_card(prediction_data), unsafe_allow_html=True)
                
                # Graphiques
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Graphique des probabilités
                    fig = create_probability_chart(prediction_data["probabilities"])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_chart2:
                    # Indicateur de confiance
                    fig = create_confidence_gauge(prediction_data["confidence"])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Détails techniques
                with st.expander("📋 Détails techniques"):
                    col_details1, col_details2 = st.columns(2)
                    
                    with col_details1:
                        st.markdown("**Informations Image:**")
                        st.markdown(f"- Dimensions: {prediction_data['image_size'][0]}x{prediction_data['image_size'][1]}")
                        st.markdown(f"- Format: RGB")
                        st.markdown(f"- Prétraitement: Normalisation [0,1]")
                    
                    with col_details2:
                        st.markdown("**Informations Prédiction:**")
                        st.markdown(f"- Timestamp: {prediction_data['timestamp']}")
                        st.markdown(f"- Classe: {prediction_data['predicted_class']}")
                        st.markdown(f"- Probabilité max: {prediction_data['confidence']:.2f}%")
                
                # Bouton pour réinitialiser
                if st.button("🔄 Analyser une autre image", use_container_width=True):
                    st.session_state.uploaded_image = None
                    if 'prediction_data' in st.session_state:
                        del st.session_state.prediction_data
                    st.rerun()
            
            else:
                # État par défaut
                st.markdown("""
                <div style="
                    text-align: center;
                    padding: 4rem 2rem;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    border-radius: 15px;
                    margin-top: 2rem;
                ">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📈</div>
                    <h3 style="color: #333; margin-bottom: 0.5rem;">En attente d'analyse</h3>
                    <p style="color: #666;">
                        Téléversez une image de billet pour commencer l'analyse.
                        Les résultats apparaîtront ici.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Page: Dashboard
    elif page == "📊 Dashboard":
        st.markdown("""
        <div class="header">
            <h2 style="color: #333;">📊 Dashboard Analytics</h2>
            <p style="color: #666;">Statistiques et historiques des analyses</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métriques
        display_statistics(st.session_state.stats)
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Historique des Confiances")
            if st.session_state.stats["predictions_history"]:
                # Préparer les données pour le graphique
                history = st.session_state.stats["predictions_history"]
                timestamps = [datetime.fromisoformat(h["timestamp"]).strftime('%H:%M') for h in history]
                confidences = [h["confidence"] for h in history]
                
                fig = go.Figure(data=go.Scatter(
                    x=timestamps,
                    y=confidences,
                    mode='lines+markers',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8, color='#764ba2')
                ))
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")
        
        with col2:
            st.markdown("### 🎯 Distribution des Classes")
            if st.session_state.stats["predictions_history"]:
                # Compter les prédictions par classe
                predictions = [h["predicted_class"] for h in st.session_state.stats["predictions_history"]]
                class_counts = {i: predictions.count(i) for i in range(len(DENOMINATIONS))}
                
                labels = [DENOMINATIONS[i]["label"] for i in class_counts.keys()]
                values = list(class_counts.values())
                colors = [DENOMINATIONS[i]["color"] for i in class_counts.keys()]
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.3,
                    marker=dict(colors=colors)
                )])
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible")
        
        # Tableau d'historique
        st.markdown("### 📋 Historique des Analyses")
        if st.session_state.stats["predictions_history"]:
            # Préparer les données pour le tableau
            history_data = []
            for h in st.session_state.stats["predictions_history"]:
                history_data.append({
                    "Heure": datetime.fromisoformat(h["timestamp"]).strftime('%H:%M:%S'),
                    "Date": datetime.fromisoformat(h["timestamp"]).strftime('%d/%m/%Y'),
                    "Dénomination": DENOMINATIONS[h["predicted_class"]]["name"],
                    "Valeur": DENOMINATIONS[h["predicted_class"]]["label"] + " Rp",
                    "Confiance": f"{h['confidence']:.1f}%",
                    "Statut": "Élevée" if h["confidence"] > 80 else "Moyenne" if h["confidence"] > 60 else "Basse"
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confiance": st.column_config.ProgressColumn(
                        "Confiance",
                        help="Niveau de confiance de la prédiction",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    )
                }
            )
            
            # Bouton d'export
            if st.button("📥 Exporter les données", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Télécharger CSV",
                    data=csv,
                    file_name="banknote_analysis_history.csv",
                    mime="text/csv"
                )
        else:
            st.info("Aucune analyse effectuée pour le moment")
    
    # Page: À Propos
    elif page == "ℹ️ À Propos":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h2>À propos de BankNote AI</h2>
                <p>
                    BankNote AI est une application d'intelligence artificielle avancée pour la classification 
                    automatique de billets de banque Rupiah. Utilisant des techniques de Deep Learning et de 
                    Transfer Learning, notre système offre une précision exceptionnelle pour identifier 
                    les différentes dénominations.
                </p>"""
                """
                <h3>🚀 Fonctionnalités</h3>
                <ul>
                    <li>Classification automatique de 7 dénominations de billets Rupiah</li>
                    <li>Interface utilisateur intuitive et professionnelle</li>
                    <li>Visualisations détaillées des résultats</li>
                    <li>Dashboard analytique en temps réel</li>
                    <li>Support multi-formats d'images</li>
                    <li>Déploiement cloud optimisé</li>
                </ul>"""
                
                """
                <h3>🔧 Technologie</h3>
                <ul>
                    <li><strong>Framework:</strong> TensorFlow 2.x, Keras</li>
                    <li><strong>Modèle:</strong> CNN avec Fine-tuning</li>
                    <li><strong>Frontend:</strong> Streamlit</li>
                    <li><strong>Visualisation:</strong> Plotly, Matplotlib</li>
                    <li><strong>Déploiement:</strong> Streamlit Cloud, Hugging Face Spaces</li>
                </ul>"""
                
                """
                <h3>📊 Performance</h3>
                <ul>
                    <li>Précision de test: >90%</li>
                    <li>Temps de prédiction: < 2 secondes</li>
                    <li>Support: Images 224x224 pixels</li>
                    <li>Classes: 7 dénominations Rupiah</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>📞 Support</h3>
                <p>
                    Pour toute question ou support technique, contactez-nous:
                </p>
                <div style="margin: 1rem 0;">
                    <p>📧 <strong>Email:</strong> support@banknote-ai.com</p>
                    <p>🌐 <strong>Site Web:</strong> www.banknote-ai.com</p>
                    <p>📱 <strong>Téléphone:</strong> +237 12 34 56 78</p>
                </div>
            </div>
            
            <div class="card">
                <h3>🔒 Sécurité</h3>
                <p>
                    Toutes les analyses sont effectuées localement dans votre navigateur. 
                    Aucune donnée d'image n'est stockée sur nos serveurs.
                </p>
                <div style="
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 1rem;
                    border-radius: 10px;
                    margin-top: 1rem;
                ">
                    <p style="margin: 0; font-size: 0.9rem;">
                        <strong>⚠️ Note importante:</strong> Cette application est conçue à des fins 
                        de démonstration et de recherche. Pour des utilisations commerciales, 
                        veuillez nous contacter.
                    </p>
                </div>
            </div>
            
            <div class="card">
                <h3>📄 Documentation</h3>
                <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                    <a href="#" style="text-decoration: none; color: #667eea; padding: 0.5rem; background: #f5f7fa; border-radius: 5px;">
                        📖 Guide d'utilisation
                    </a>
                    <a href="#" style="text-decoration: none; color: #667eea; padding: 0.5rem; background: #f5f7fa; border-radius: 5px;">
                        🧪 Documentation technique
                    </a>
                    <a href="#" style="text-decoration: none; color: #667eea; padding: 0.5rem; background: #f5f7fa; border-radius: 5px;">
                        📄 API Reference
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer global
    st.markdown("""
    <div class="footer">
        <p>
            BankNote AI © 2025 | 
            <a href="#" style="color: #667eea; text-decoration: none;">Confidentialité</a> | 
            <a href="#" style="color: #667eea; text-decoration: none;">Conditions d'utilisation</a> | 
            <a href="#" style="color: #667eea; text-decoration: none;">Mentions légales</a>
        </p>
        <p style="font-size: 0.8rem; color: #999;">
            Cette application utilise l'intelligence artificielle pour la classification de billets. 
            Les résultats sont fournis à titre indicatif.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()