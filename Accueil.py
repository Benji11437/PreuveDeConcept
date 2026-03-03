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

# --- CSS moderne ---
st.markdown("""
<style>
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
    color: #555;
}

.hero img {
    margin-top: 40px;
    width: 65%;
    border-radius: 20px;
    box-shadow: 0px 15px 40px rgba(0,0,0,0.15);
    transition: transform 0.3s ease;
}

.hero img:hover {
    transform: scale(1.03);
}

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

# --- Hero Section ---
st.markdown(f"""
<div class="hero">

<h1>Analyse des sentiments</h1>

<p>
Bienvenue dans l'application d'analyse de sentiments.
Prédiction du sentiment des tweets en temps réel.
</p>

<img src="data:image/jpg;base64,{img_base64}">

</div>
""", unsafe_allow_html=True)

# --- Section fonctionnalités ---
st.markdown("""
<div class="features">

### Fonctionnalités principales

<div class="feature-box">
📊 <b>Analyse exploratoire</b> – Visualisez les distributions et tendances  
🤖 <b>Prédiction automatique</b> – Détection Positif / Négatif  
☁️ <b>Analyse lexicale</b> – WordCloud et mots fréquents  
</div>

</div>
""", unsafe_allow_html=True)