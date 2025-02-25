import streamlit as st
import pandas as pd
import os
import requests
from datetime import datetime
from transformers import pipeline

# ---------------------- INITIALISATION ---------------------- #

def load_bearer_token():
    bearer_token = st.secrets.get("BEARER_TOKEN")
    if not bearer_token:
        st.error("❌ Bearer Token manquant. Vérifiez vos secrets Streamlit.")
    return bearer_token

sentiment_analyzer = pipeline("sentiment-analysis")

if "tweets" not in st.session_state:
    st.session_state.tweets = pd.DataFrame(columns=["Date", "Utilisateur", "Texte", "Sentiment", "Score"])

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], round(result['score'] * 100, 2)

def save_tweets_to_csv():
    st.session_state.tweets.to_csv("tweets.csv", index=False)

def load_tweets_from_csv():
    if os.path.exists("tweets.csv"):
        st.session_state.tweets = pd.read_csv("tweets.csv")

load_tweets_from_csv()

# ---------------------- COLLECTE DE TWEETS (API V2) ---------------------- #

def collect_recent_tweets(bearer_token, keywords, max_results=20):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    query = " OR ".join([k.strip() for k in keywords if k.strip()]) + " -is:retweet lang:fr"
    
    url = (
        f"https://api.twitter.com/2/tweets/search/recent?"
        f"query={query}&max_results={max_results}&tweet.fields=author_id,created_at"
    )

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.error(f"🚫 Erreur lors de la collecte des tweets : {response.status_code} - {response.json()}")
        return

    data = response.json()
    tweets = data.get("data", [])

    if not tweets:
        st.info("🔍 Aucun tweet trouvé pour les mots-clés fournis.")
        return

    new_tweets = []
    for tweet in tweets:
        sentiment, score = analyze_sentiment(tweet["text"])
        new_tweets.append({
            "Date": tweet["created_at"],
            "Utilisateur": tweet["author_id"],
            "Texte": tweet["text"],
            "Sentiment": sentiment,
            "Score": score
        })

    st.session_state.tweets = pd.concat(
        [pd.DataFrame(new_tweets), st.session_state.tweets],
        ignore_index=True
    ).drop_duplicates(subset=["Texte"])

    save_tweets_to_csv()
    st.success(f"✅ {len(new_tweets)} tweets collectés avec succès.")

# ---------------------- INTERFACE STREAMLIT ---------------------- #

st.title("🐦 Agent Twitter AI - Dashboard (Version Gratuite API v2)")

keywords_input = st.text_input("🔎 Mots-clés à rechercher (séparés par des virgules):", "cryptomonnaie, blockchain, web3, politique, technologies")
keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

max_results = st.slider("Nombre de tweets à collecter par recherche :", min_value=5, max_value=50, value=20, step=5)

if st.button("📥 Collecter les tweets"):
    bearer_token = load_bearer_token()
    if bearer_token:
        collect_recent_tweets(bearer_token, keywords, max_results)

st.markdown("---")

st.subheader("📄 Tweets collectés")
if not st.session_state.tweets.empty:
    st.dataframe(st.session_state.tweets.head(10))
else:
    st.info("Aucun tweet collecté pour le moment.")

st.markdown("---")

st.subheader("📊 Visualisation des sentiments")
if not st.session_state.tweets.empty:
    sentiment_counts = st.session_state.tweets["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)
else:
    st.info("Aucune donnée disponible pour les graphiques.")

st.markdown("---")

st.subheader("💾 Exporter les données")
if not st.session_state.tweets.empty:
    csv = st.session_state.tweets.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les tweets en CSV",
        data=csv,
        file_name="tweets_collected.csv",
        mime="text/csv",
    )
else:
    st.info("Collectez des tweets avant de pouvoir les télécharger.")
