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
# CONFIGURATION DE LA PAGE
# ===============================
st.set_page_config(page_title="Dashboard Tweets", layout="wide")

st.title("📊 Dashboard – Analyse descriptive des tweets")
st.markdown(
    "Analyse exploratoire des tweets : structure du jeu de données, "
    "distribution des sentiments et analyse lexicale."
)

# ===============================
# CHARGEMENT DES DONNÉES
# ===============================
df = pd.read_csv("data/df_c.csv")
df = df.rename(columns={"lemmatize_joined": "text"})

# Sécurisation
df["text"] = df["text"].fillna("").astype(str)

# ===============================
# APERÇU DES DONNÉES
# ===============================
st.subheader("🔍 Aperçu des données")
st.dataframe(df.head(), use_container_width=True)

# ===============================
# KPIs GLOBAUX
# ===============================
st.subheader("Indicateurs clés")

col1, col2, col3 = st.columns(3)

col1.metric("Nombre total de tweets", len(df))
col2.metric(
    "Longueur moyenne (caractères)",
    round(df["text"].str.len().mean(), 1)
)
col3.metric("Nombre de catégories", df["target"].nunique())


# ===============================
# PRÉTRAITEMENT NLP
# ===============================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

df["nb_words"] = df["text"].apply(lambda x: len(x.split()))

def clean_tokens(text):
    return [
        w.lower()
        for w in text.split()
        if w.isalpha() and w.lower() not in stop_words
    ]

tokens = df["text"].apply(clean_tokens).sum()
word_freq = Counter(tokens).most_common(20)
top_words = [w for w, _ in word_freq]

# ===============================
# ANALYSE TEXTUELLE – ONGLET
# ===============================
st.subheader("Analyse textuelle détaillée")

tab1, tab2, tab3 = st.tabs(
    ["📈 Longueur des tweets", "📊 Lexique & mots", "☁️ WordCloud"]
)

# ---------- ONGLET 1
with tab1:
    st.markdown("Distribution du nombre de mots par tweet.")

    fig_hist = px.histogram(
        df,
        x="nb_words",
        nbins=20,
        labels={"nb_words": "Nombre de mots"},
        title="Distribution de la longueur des tweets"
    )
    fig_hist.update_layout(
        template="plotly_white",
        xaxis_title="Nombre de mots",
        yaxis_title="Nombre de tweets"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_box = px.box(
        df,
        x="target",
        y="nb_words",
        color="target",
        labels={"nb_words": "Nombre de mots"},
        title="Longueur des tweets par sentiment",
        color_discrete_map={
            "Positif": "#1b9e77",
            "Négatif": "#d95f02"
        }
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ---------- ONGLET 2
with tab2:
    st.markdown("Mots les plus fréquents après suppression des stopwords.")

    df_words = pd.DataFrame(word_freq, columns=["Mot", "Fréquence"])

    chart_bar = alt.Chart(df_words).mark_bar().encode(
        x=alt.X("Fréquence:Q", title="Fréquence"),
        y=alt.Y("Mot:N", sort="-x", title="Mot"),
        tooltip=["Mot", "Fréquence"]
    ).properties(
        title="Top 20 des mots les plus fréquents"
    )

    st.altair_chart(chart_bar, use_container_width=True)

# ---------- ONGLET 3
with tab3:
    st.markdown(
        "Nuage de mots représentant visuellement les termes les plus fréquents."
    )

    wc = WordCloud(
        background_color="white",
        colormap="viridis",
        width=700,
        height=400
    ).generate(" ".join(tokens))

    fig_wc, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig_wc)
