import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Analyse des tweets",
    layout="wide"
)


st.title(" Analyse des sentiments")

# Charger l'image
image = Image.open("futur.jpg")

st.markdown("""
Bienvenue dans l'application d'analyse de sentiments.

Utilisez le menu à gauche pour naviguer :
- **Analyse exploratoire**
- **Prédiction de sentiment**
""")


# Afficher l'image
# Image centrée


# st.image("futur.jpg", width=700)

st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("futur.jpg", width=700)
st.markdown("</div>", unsafe_allow_html=True)
