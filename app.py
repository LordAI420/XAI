import streamlit as st
import pandas as pd
import threading
import time
import tweepy
import os
from datetime import datetime
from transformers import pipeline

# ---------------------- INITIALISATION ---------------------- #

def load_twitter_api():
    consumer_key = st.secrets["API_KEY"]
    consumer_secret = st.secrets["API_SECRET"]
    access_token = st.secrets["ACCESS_TOKEN"]
    access_token_secret = st.secrets["ACCESS_TOKEN_SECRET"]

    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

sentiment_analyzer = pipeline("sentiment-analysis")

# Initialisation des états Streamlit
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False

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

# ---------------------- STREAMING ---------------------- #
class MyStreamListener(tweepy.Stream):
    def on_status(self, status):
        if hasattr(status, "retweeted_status") or status.lang != "fr":
            return

        sentiment, score = analyze_sentiment(status.text)
        new_tweet = {
            "Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Utilisateur": status.user.screen_name,
            "Texte": status.text,
            "Sentiment": sentiment,
            "Score": score
        }
        st.session_state.tweets = pd.concat([pd.DataFrame([new_tweet]), st.session_state.tweets], ignore_index=True)
        save_tweets_to_csv()

    def on_error(self, status_code):
        if status_code == 420:
            st.error("🚫 Limite de taux atteinte. L'agent va se mettre en pause.")
            return False

def collect_tweets(api, keywords):
    stream = MyStreamListener(api.auth.consumer_key, api.auth.consumer_secret, api.auth.access_token, api.auth.access_token_secret)
    threading.Thread(target=stream.filter, kwargs={'track': keywords, 'languages': ["fr"], 'is_async': True}).start()
    return stream

# ---------------------- INTERFACE STREAMLIT ---------------------- #

st.title("🐦 Agent Twitter AI - Dashboard")

keywords_input = st.text_input("🔎 Mots-clés à suivre (séparés par des virgules):", "cryptomonnaie, blockchain, web3, politique, technologies")
keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Démarrer l'agent"):
        if not st.session_state.agent_running:
            api = load_twitter_api()
            st.session_state.stream = collect_tweets(api, keywords)
            st.session_state.agent_running = True
            st.success("✅ Agent démarré.")
        else:
            st.warning("⚠️ L'agent est déjà en cours d'exécution.")

with col2:
    if st.button("⏹️ Arrêter l'agent"):
        if st.session_state.agent_running and hasattr(st.session_state, "stream"):
            st.session_state.stream.disconnect()
            st.session_state.agent_running = False
            st.success("🛑 Agent arrêté.")
        else:
            st.warning("⚠️ Aucun agent en cours d'exécution.")

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
