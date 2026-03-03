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

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(image)