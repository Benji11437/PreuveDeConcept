import streamlit as st
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import stopwords
import plotly.express as px
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ===============================
# CONFIGURATION
# ===============================
st.set_page_config(page_title="Dashboard Tweets", layout="wide")

st.title("Dashboard – Analyse descriptive des tweets")

# ===============================
# TELECHARGEMENT STOPWORDS (une seule fois)
# ===============================
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = load_stopwords()

# ===============================
# CHARGEMENT DONNÉES (cache)
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/df_ope.csv")
    df = df.rename(columns={"lemm_text": "text"})
    df["text"] = df["text"].fillna("").astype(str)
    return df

df = load_data()

# ===============================
# APERÇU
# ===============================
st.subheader(" Aperçu des données")
st.dataframe(df.head(), use_container_width=True)

# ===============================
# KPIs
# ===============================
st.subheader("Indicateurs clés")

col1, col2, col3 = st.columns(3)

col1.metric("Nombre total de tweets", len(df))
col2.metric("Longueur moyenne (caractères)", round(df["text"].str.len().mean(), 1))
col3.metric("Nombre de catégories", df["target"].nunique())

# ===============================
# PRÉTRAITEMENT OPTIMISÉ (cache)
# ===============================
@st.cache_data
def preprocess_text(dataframe):

    # Nombre de mots (vectorisé → plus rapide)
    dataframe["nb_words"] = dataframe["text"].str.split().str.len()

    # Nettoyage rapide
    tokens = []
    for text in dataframe["text"]:
        for w in text.split():
            w = w.lower()
            if w.isalpha() and w not in stop_words:
                tokens.append(w)

    word_freq = Counter(tokens).most_common(20)

    return dataframe, tokens, word_freq

df, tokens, word_freq = preprocess_text(df)

# ===============================
# ANALYSE TEXTUELLE
# ===============================
st.subheader("Analyse textuelle détaillée")

tab1, tab2, tab3 = st.tabs(
    ["📈 Longueur des tweets", "📊 Lexique & mots", "☁️ WordCloud"]
)

# ---------- ONGLET 1
with tab1:

    fig_hist = px.histogram(
        df,
        x="nb_words",
        nbins=20,
        title="Distribution de la longueur des tweets"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_box = px.box(
        df,
        x="target",
        y="nb_words",
        color="target",
        title="Longueur des tweets par sentiment",
        color_discrete_map={
            "Positif": "#1b9e77",
            "Négatif": "#d95f02"
        }
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ---------- ONGLET 2
with tab2:

    df_words = pd.DataFrame(word_freq, columns=["Mot", "Fréquence"])

    chart_bar = alt.Chart(df_words).mark_bar().encode(
        x="Fréquence:Q",
        y=alt.Y("Mot:N", sort="-x"),
        tooltip=["Mot", "Fréquence"]
    )

    st.altair_chart(chart_bar, use_container_width=True)

# ---------- ONGLET 3
with tab3:

    @st.cache_data
    def generate_wordcloud(tokens):
        return WordCloud(
            background_color="white",
            width=700,
            height=400
        ).generate(" ".join(tokens))

    wc = generate_wordcloud(tokens)

    fig_wc, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig_wc)