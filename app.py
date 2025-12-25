import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertForSequenceClassification
import matplotlib.pyplot as plt

# ===============================
# CONFIGURATION PAGE
# ===============================
st.set_page_config(
    page_title="Preuve de concept – Analyse des sentiments",
    layout="centered"
)

# ===============================
# CHARGEMENT DU MODÈLE
# ===============================
@st.cache_resource
def load_model():
    MODEL_PATH = "modernBERT_twitter_sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = ModernBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ===============================
# TITRE & CONTEXTE
# ===============================
st.title("Analyse des sentiments - Tweets")

st.markdown("""
### Objectif du projet  
Cette preuve de concept montre la capacité d’un modèle NLP (Transformers)
à prédire le sentiment d’un tweet (Positif ou Négatif) pour
aider à l’analyse d’une opinion.
""")

# ===============================
# PRÉDICTION EN TEMPS RÉEL
# ===============================
st.header(" Prédiction en temps réel")
tweet = st.text_area(
    "Saisir le tweet ici :",
    height=120,
    placeholder="Exemple : I really love this new product!"
)

if st.button(" Prédire le sentiment") and tweet.strip():

    # Tokenization
    inputs = tokenizer(
        tweet,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    # Prédiction + attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions  # tuple [num_layers, batch, num_heads, seq_len, seq_len]
        probs = F.softmax(logits, dim=1).squeeze()
        pred_class = int(torch.argmax(probs).item())

    labels = {0: "Négatif", 1: "Positif"}
    sentiment = labels[pred_class]

    # Affichage du résultat
    st.subheader("Résultat de la prédiction")
    col1, col2 = st.columns(2)
    with col1:
        if pred_class == 1:
            st.success(f"✅ Sentiment détecté : **{sentiment}**")
        else:
            st.error(f"⚠️ Sentiment détecté : **{sentiment}**")
    with col2:
        st.metric("Probabilité Positif", f"{probs[1]*100:.2f}%")
        st.metric("Probabilité Négatif", f"{probs[0]*100:.2f}%")

    
# ===============================
# INTERPRÉTATION MÉTIER
# ===============================
st.header("🧠 Interprétation métier")
st.markdown("""
- Le modèle permet d’analyser automatiquement le sentiment exprimé dans un tweet.
- Utile pour :
  - veille de marque,
  - analyse de satisfaction client,
  - détection de tendances d’opinion.
""")

# ===============================
# LIMITES & PERSPECTIVES
# ===============================
st.header("Limites et perspectives")
st.markdown("""
**Limites :**
- Binaire (Positif / Négatif)
- Sensible à l’ironie / sarcasme
- Dépend des données d’entraînement

**Perspectives :**
- Ajouter une classe Neutre
- Analyse multilingue
- Déploiement cloud pour usage métier
- Analyse en batch de plusieurs tweets
""")
