import streamlit as st
import base64

st.set_page_config(
    page_title="Analyse des Tweets",
    layout="wide"
)

# --- Encoder l'image en base64 ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("futur.jpg")

# --- CSS accessible (WCAG) ---
st.markdown("""
<style>

/* Focus visible pour navigation clavier (WCAG 2.4.7) */
button:focus, a:focus {
    outline: 3px solid #005fcc;
    outline-offset: 2px;
}

/* Structure principale */
.hero {
    text-align: center;
    padding: 60px 20px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 10px;
}

.hero p {
    font-size: 20px;
    color: #333; /* contraste amélioré */
}

/* Image principale */
.hero img {
    margin-top: 40px;
    width: 65%;
    border-radius: 20px;
    box-shadow: 0px 15px 40px rgba(0,0,0,0.15);
}

/* Section fonctionnalités */
.features {
    text-align: center;
    margin-top: 60px;
    font-size: 18px;
}

.feature-box {
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# --- Lien d'accès rapide (WCAG 2.4.1 Skip navigation) ---
st.markdown(
"""
<a href="#main-content" style="position:absolute;left:-999px;">
Passer directement au contenu principal
</a>
""",
unsafe_allow_html=True
)

# --- Hero Section ---
st.markdown(f"""
<div class="hero">

<h1>Analyse des sentiments</h1>

<p>
    Bienvenue sur le dashboard de prédiction des sentiments.
    Cette plateforme permet de prédire automatiquement, en temps réel, le sentiment des tweets grâce à un modèle de traitement du langage naturel.
</p>

<img src="data:image/jpg;base64,{img_base64}"
     alt="Illustration représentant l’analyse automatique des sentiments sur des tweets">

</div>
""", unsafe_allow_html=True)

# --- Contenu principal ---
st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)

# --- Section fonctionnalités ---
st.markdown("""
<div class="features">

<h2>Fonctionnalités principales</h2>

<div class="feature-box">

<p>
📊 <strong>Analyse exploratoire</strong> – Visualisation des distributions et tendances des tweets.
</p>

<p>
  <strong>Prédiction automatique</strong> – Détection du sentiment positif ou négatif à l’aide d’un modèle NLP.
</p>

<p>
☁️ <strong>Analyse lexicale</strong> – Visualisation des mots fréquents et génération d’un nuage de mots.
</p>

</div>

</div>
""", unsafe_allow_html=True)

# --- Déclaration accessibilité ---
st.markdown("""
### Accessibilité

Le contenu de cette page respecte plusieurs recommandations des **WCAG 2.2 – niveau AA** :

- présence d'alternatives textuelles pour les images
- structure claire des titres
- contraste suffisant pour le texte
- navigation possible au clavier
- lien d'accès rapide au contenu principal

Ces mesures améliorent l’accessibilité pour les utilisateurs
de lecteurs d’écran ou ayant des déficiences visuelles ou motrices.
""")