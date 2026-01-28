import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, ModernBertForSequenceClassification
import matplotlib.pyplot as plt

# ===============================
# CONFIGURATION DE LA PAGE
# ===============================
st.set_page_config(
    page_title="Analyse des sentiments",
    layout="centered"
)

# ===============================
# CHARGEMENT DU MODÈLE
# ===============================
@st.cache_resource
def load_model():
    MODEL_PATH = "Benji1437/modernbert-twitter-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = ModernBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ===============================
# TITRE & CONTEXTE
# ===============================
st.title("Analyse des sentiments")

st.markdown("""
Cette page permet d’analyser automatiquement le **sentiment d’un tweet**
(**positif ou négatif**) à l’aide d’un modèle NLP basé sur Transformers.
""")

# ===============================
# SAISIE UTILISATEUR
# ===============================
st.header("Prédiction en temps réel")

tweet = st.text_area(
    label="Texte du tweet à analyser",
    height=120,
    placeholder="Exemple : I really love this new product!",
    help="Saisissez un tweet en anglais."
)

# ===============================
# ACTION UTILISATEUR
# ===============================
if st.button("Prédire le sentiment"):

    if not tweet.strip():
        st.warning("Veuillez saisir un texte avant de lancer la prédiction.")
    else:
        inputs = tokenizer(
            tweet,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze()
            pred_class = int(torch.argmax(probs).item())

        labels = {0: "Négatif", 1: "Positif"}
        sentiment = labels[pred_class]

        st.subheader("Résultat")

        col1, col2 = st.columns(2)

        with col1:
            if pred_class == 1:
                st.success(f"Sentiment détecté : **{sentiment}**")
            else:
                st.error(f"Sentiment détecté : **{sentiment}**")

        with col2:
            st.metric("Probabilité Positif", f"{probs[1]*100:.2f}%")
            st.metric("Probabilité Négatif", f"{probs[0]*100:.2f}%")

        df_probs = pd.DataFrame({
            "Sentiment": ["Positif", "Négatif"],
            "Probabilité (%)": [
                float(probs[1] * 100),
                float(probs[0] * 100)
            ]
        })

        st.subheader("Visualisation")

        fig, ax = plt.subplots()
        ax.bar(df_probs["Sentiment"], df_probs["Probabilité (%)"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probabilité (%)")

        for i, value in enumerate(df_probs["Probabilité (%)"]):
            ax.text(i, value + 1, f"{value:.1f}%", ha="center")

        st.pyplot(fig)

        st.subheader("Détails numériques")
        st.dataframe(df_probs, use_container_width=True)

