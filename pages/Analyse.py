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
# Page configuration
# ===============================
st.set_page_config(page_title="Dashboard Tweets", layout="wide")
st.title("Analyse exploratoire des tweets")

# ===============================
# Chargement des données
# ===============================

df = pd.read_csv("data/df_c.csv") 
df = df.rename(columns={"lemmatize_joined": "text"})


# ===============================
# Prétraitement
# ===============================
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

df["nb_words"] = df["text"].apply(lambda x: len(x.split()))

def clean_tokens(text):
    return [w.lower() for w in text.split() if w.isalpha() and w.lower() not in stop_words]

tokens = df["text"].apply(clean_tokens).sum()
word_freq = Counter(tokens).most_common(20)
top_words = [w for w, _ in word_freq]

# ===============================
# Onglets accessibles
# ===============================
tab1, tab2, tab3 = st.tabs(["Longueur des tweets", "Lexique & mots", "WordCloud"])

# ---------- Onglet 1 : Longueur des tweets ----------
with tab1:
    st.subheader("Distribution de la longueur des tweets")
    fig_hist = px.histogram(df, x="nb_words", nbins=20,
                            labels={"nb_words": "Nombre de mots"},
                            title="Histogramme interactif - Longueur des tweets",
                            color_discrete_sequence=["#1f77b4"])  # couleur accessible daltoniens
    fig_hist.update_layout(
        template="plotly_white",
        xaxis_title="Nombre de mots par tweet",
        yaxis_title="Nombre de tweets"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Longueur des tweets par sentiment")
    fig_box = px.box(df, x="target", y="nb_words",
                     labels={"nb_words": "Nombre de mots"},
                     title="Boxplot interactif - Longueur par sentiment",
                     color="target",
                     color_discrete_map={"Positif":"#2ca02c","Négatif":"#d62728"})  # contraste élevé
    st.plotly_chart(fig_box, use_container_width=True)

# ---------- Onglet 2 : Lexique & mots ----------
with tab2:
    st.subheader("Top 20 des mots les plus fréquents")
    df_words = pd.DataFrame(word_freq, columns=["word", "count"])
    chart_bar = alt.Chart(df_words).mark_bar().encode(
        x=alt.X("count:Q", title="Fréquence"),
        y=alt.Y("word:N", sort="-x", title="Mot"),
        tooltip=["word:N", "count:Q"]
    ).properties(title="Bar chart interactif - Top 20 mots")
    st.altair_chart(chart_bar, use_container_width=True)

    st.subheader("Fréquence des mots par sentiment")
    rows = []
    for sentiment in df["target"].unique():
        subset = df[df["target"] == sentiment]["text"]
        words = subset.apply(clean_tokens).sum()
        counts = Counter(words)
        for w in top_words:
            rows.append({"word": w, "target": sentiment, "count": counts.get(w, 0)})
    df_heat = pd.DataFrame(rows)
    heatmap = alt.Chart(df_heat).mark_rect().encode(
        x="target:N",
        y=alt.Y("word:N", sort="-x"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Fréquence"),
        tooltip=["word:N", "target:N", "count:Q"]
    ).properties(title="Heatmap interactif - Mots par sentiment")
    st.altair_chart(heatmap, use_container_width=True)

# ---------- Onglet 3 : WordCloud ----------
with tab3:
    st.subheader("WordCloud des tweets")
    st.write("Description alternative : nuage de mots montrant les termes les plus fréquents dans les tweets analysés.")
    wc = WordCloud(background_color="white", colormap="winter",
                   width=600, height=400).generate(" ".join(tokens))
    fig_wc, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)
