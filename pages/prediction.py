import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator

# ===============================
# CONFIGURATION DE LA PAGE
# ===============================
st.set_page_config(
    page_title="Analyse des sentiments – Tweets",
    layout="centered"
)

# ===============================
# CHARGEMENT DU MODÈLE DE SENTIMENT
# ===============================
@st.cache_resource
def load_models():
    SENTIMENT_MODEL_PATH = "Benji1437/modernbert-twitter-sentiment"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
    sentiment_model.eval()
    return sentiment_tokenizer, sentiment_model

sentiment_tokenizer, sentiment_model = load_models()

# ===============================
# FONCTION DE TRADUCTION
# ===============================
def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        st.error(f"Erreur lors de la traduction : {e}")
        return text  # fallback : analyse du texte original

# ===============================
# CHARGEMENT DES DONNÉES
# ===============================
df = pd.read_csv("data/df_ope.csv")
df = df.rename(columns={"lemm_text": "text"})
df["text"] = df["text"].fillna("").astype(str)
df["Name"] = df["Name"].fillna("Inconnu").astype(str)

# ===============================
# TITRE & CONTEXTE
# ===============================
st.title("Analyse des sentiments – Tweets")

st.markdown(
    "Cette page permet d’analyser automatiquement le **sentiment d’un tweet** "
    "(**positif ou négatif**) à l’aide d’un modèle NLP basé sur Transformers.\n\n"
    " Les tweets saisis manuellement sont **automatiquement traduits en anglais** avant l’analyse."
)

# ===============================
# CHOIX DE LA SOURCE DU TWEET
# ===============================
st.header("Prédiction en temps réel")

mode = st.radio(
    "Choisissez la source du tweet :",
    ["Saisie manuelle", "Sélection depuis la base de données"]
)

# ===============================
# SAISIE / SÉLECTION DU TWEET
# ===============================
translated_tweet = None

if mode == "Saisie manuelle":
    original_tweet = st.text_area(
        label="Texte du tweet à analyser (toutes langues)",
        height=120,
        placeholder=" ",
        help="Le texte sera automatiquement traduit en anglais."
    )
    name = "Saisie manuelle"

else:
    # Échantillon stable pour la sélection
    if "tweet_sample_df" not in st.session_state:
        st.session_state.tweet_sample_df = (
            df[["text", "Name"]]
            .drop_duplicates()
            .sample(min(100, len(df)), random_state=42)
            .reset_index(drop=True)
        )

    sample_df = st.session_state.tweet_sample_df

    selected_index = st.selectbox(
        "Sélectionnez un tweet existant :",
        options=sample_df.index,
        format_func=lambda i: sample_df.loc[i, "text"],
        help="Sélection par contenu du tweet."
    )

    original_tweet = sample_df.loc[selected_index, "text"]
    name = sample_df.loc[selected_index, "Name"]

    st.caption(
        f"""
**Auteur (Name)** : {name}

**Tweet sélectionné :**  
> {original_tweet}
"""
    )

# ===============================
# ACTION UTILISATEUR
# ===============================
if st.button(" Prédire le sentiment"):

    if not original_tweet.strip():
        st.warning("Veuillez saisir ou sélectionner un tweet avant de lancer la prédiction.")
    else:
        # Traduction si saisie manuelle
        if mode == "Saisie manuelle":
            translated_tweet = translate_to_english(original_tweet)
            tweet_for_analysis = translated_tweet
        else:
            tweet_for_analysis = original_tweet

        # Tokenisation et prédiction
        inputs = sentiment_tokenizer(
            tweet_for_analysis,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze()
            pred_class = int(torch.argmax(probs).item())

        labels = {0: "Négatif", 1: "Positif"}
        sentiment = labels[pred_class]

        # ===============================
        # AFFICHAGE DES RÉSULTATS
        # ===============================
        st.subheader("Résultat de la prédiction")

        st.markdown(f"**Auteur (Name)** : {name}")
        st.markdown(f"**Tweet original :**\n> {original_tweet}")

        if translated_tweet:
            st.markdown(f"**Tweet traduit (anglais) :**\n> {translated_tweet}")

        col1, col2 = st.columns(2)

        with col1:
            if pred_class == 1:
                st.success(f"Sentiment détecté : **{sentiment}**")
            else:
                st.error(f"Sentiment détecté : **{sentiment}**")

        with col2:
            st.metric("Probabilité Positif", f"{probs[1]*100:.2f}%")
            st.metric("Probabilité Négatif", f"{probs[0]*100:.2f}%")

        # Visualisation graphique
        df_probs = pd.DataFrame({
            "Sentiment": ["Positif", "Négatif"],
            "Probabilité (%)": [
                float(probs[1] * 100),
                float(probs[0] * 100)
            ]
        })

        st.subheader(" Visualisation des probabilités")
        fig, ax = plt.subplots()
        ax.bar(df_probs["Sentiment"], df_probs["Probabilité (%)"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probabilité (%)")
        for i, value in enumerate(df_probs["Probabilité (%)"]):
            ax.text(i, value + 1, f"{value:.1f}%", ha="center")
        st.pyplot(fig)

        # Détails numériques
        st.subheader("📋 Détails numériques")
        st.dataframe(df_probs, use_container_width=True)

# ===============================
# INTERPRÉTATION MÉTIER
# ===============================
st.header(" Interprétation métier")
st.markdown("""
- Les tweets saisis manuellement sont automatiquement traduits en anglais
- Sélection par tweet depuis la base de données
- Affichage automatique de l’auteur (Name)
- Chaîne complète : **Texte original → Traduction → Analyse de sentiment**
""")


# ===============================
# LIMITES & PERSPECTIVES
# ===============================
st.header(" Limites et perspectives")

st.markdown("""
**Limites :**
- Classification binaire (positif / négatif)
- Sensible à l’ironie et au sarcasme
- Dépend fortement des données d’entraînement

**Perspectives :**
- Ajout d’une classe neutre
- Analyse multilingue
- Traitement par lot (batch)
- Déploiement cloud
""")

# ===============================
# DÉCLARATION ACCESSIBILITÉ
# ===============================
st.header(" Déclaration d’accessibilité")

st.markdown("""
Cette application applique les bonnes pratiques des **WCAG 2.2 – niveau AA** :
- Navigation clavier
- Labels et instructions explicites
- Résultats compréhensibles sans dépendre de la couleur
- Graphiques accompagnés d’alternatives textuelles

**Limites connues :**
- Personnalisation ARIA limitée par Streamlit
""")
