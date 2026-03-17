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

# WCAG : titre principal clair pour la structure du document
st.title("Analyse des sentiments – Tweets")

# WCAG : description générale de la page
st.markdown(
"""
Cette application permet d’analyser automatiquement le **sentiment d’un tweet**
(**positif ou négatif**) à l’aide d’un modèle de traitement du langage naturel.

Les tweets saisis manuellement sont **automatiquement traduits en anglais**
avant l’analyse afin d’être compatibles avec le modèle utilisé.
"""
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
        return text

# ===============================
# CHARGEMENT DES DONNÉES
# ===============================
df = pd.read_csv("data/df_ope.csv")
df = df.rename(columns={"lemm_text": "text"})
df["text"] = df["text"].fillna("").astype(str)
df["Name"] = df["Name"].fillna("Inconnu").astype(str)

# ===============================
# CHOIX DE LA SOURCE DU TWEET
# ===============================
st.header("Prédiction en temps réel")

# WCAG : instruction explicite pour les utilisateurs
st.caption(
"Sélectionnez la source du tweet à analyser. Vous pouvez saisir un tweet manuellement "
"ou choisir un tweet existant dans la base de données."
)

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
        placeholder="Saisissez ici le texte du tweet",
        help="Le texte sera automatiquement traduit en anglais avant l’analyse."
    )

    # WCAG : instruction claire
    st.caption(
        "Le tweet peut être rédigé dans n'importe quelle langue. "
        "La traduction en anglais sera effectuée automatiquement."
    )

    name = "Saisie manuelle"

else:

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
        help="La sélection affiche également l’auteur associé au tweet."
    )

    original_tweet = sample_df.loc[selected_index, "text"]
    name = sample_df.loc[selected_index, "Name"]

    st.caption(
        f"""
Auteur du tweet : **{name}**

Tweet sélectionné :
> {original_tweet}
"""
    )

# ===============================
# ACTION UTILISATEUR
# ===============================
if st.button("Prédire le sentiment"):

    if not original_tweet.strip():

        # WCAG : message explicite
        st.warning(
            "Veuillez saisir un tweet ou sélectionner un tweet dans la liste avant de lancer la prédiction."
        )

    else:

        if mode == "Saisie manuelle":
            translated_tweet = translate_to_english(original_tweet)
            tweet_for_analysis = translated_tweet
        else:
            tweet_for_analysis = original_tweet

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

        st.markdown(f"**Auteur :** {name}")
        st.markdown(f"**Tweet original :**\n> {original_tweet}")

        if translated_tweet:
            st.markdown(f"**Tweet traduit en anglais :**\n> {translated_tweet}")

        col1, col2 = st.columns(2)

        with col1:

            if pred_class == 1:
                st.success(f"Sentiment détecté : {sentiment}")
            else:
                st.error(f"Sentiment détecté : {sentiment}")

            # WCAG : information textuelle indépendante de la couleur
            st.caption(
                "Le sentiment est indiqué textuellement afin de rester compréhensible "
                "même sans perception des couleurs."
            )

        with col2:

            st.metric("Probabilité Positif", f"{probs[1]*100:.2f}%")
            st.metric("Probabilité Négatif", f"{probs[0]*100:.2f}%")

        # ===============================
        # VISUALISATION
        # ===============================
        st.subheader("Visualisation des probabilités")

        st.caption(
            "Graphique représentant la probabilité prédite pour chaque sentiment."
        )

        df_probs = pd.DataFrame({
            "Sentiment": ["Positif", "Négatif"],
            "Probabilité (%)": [
                float(probs[1] * 100),
                float(probs[0] * 100)
            ]
        })

        fig, ax = plt.subplots()

        ax.bar(df_probs["Sentiment"], df_probs["Probabilité (%)"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probabilité (%)")

        for i, value in enumerate(df_probs["Probabilité (%)"]):
            ax.text(i, value + 1, f"{value:.1f}%", ha="center")

        st.pyplot(fig)

        # WCAG : alternative textuelle du graphique
        st.caption(
            "Alternative textuelle : la barre la plus élevée indique le sentiment "
            "ayant la probabilité la plus forte."
        )

        # ===============================
        # TABLEAU NUMÉRIQUE
        # ===============================
        st.subheader("Détails numériques")

        st.caption(
            "Tableau récapitulatif des probabilités associées à chaque sentiment."
        )

        st.dataframe(df_probs, use_container_width=True)

# ===============================
# INTERPRÉTATION MÉTIER
# ===============================
st.header("Interprétation métier")

st.markdown("""
- Les tweets saisis manuellement sont automatiquement traduits en anglais
- Sélection possible d’un tweet existant dans la base de données
- Affichage de l’auteur associé au tweet
- Chaîne complète : **Texte original → Traduction → Analyse de sentiment**
""")

# ===============================
# LIMITES
# ===============================
st.header("Limites et perspectives")

st.markdown("""
**Limites :**
- Classification binaire (positif / négatif)
- Difficulté à détecter l’ironie ou le sarcasme
- Dépendance aux données d’entraînement

**Perspectives :**
- Ajout d’une classe neutre
- Analyse multilingue native
- Traitement par lot
- Déploiement cloud
""")

# ===============================
# ACCESSIBILITÉ
# ===============================
st.header("Déclaration d’accessibilité")

st.markdown("""
Cette application suit les bonnes pratiques des **WCAG 2.2 – niveau AA** :

- Navigation possible au clavier
- Instructions et labels explicites
- Résultats compréhensibles sans dépendre uniquement de la couleur
- Graphiques accompagnés de descriptions textuelles
- Présence d’alternatives textuelles aux visualisations

**Limites connues :**
- Personnalisation ARIA limitée par les composants Streamlit.
""")