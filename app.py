import streamlit as st
import pandas as pd
import threading
import os
from datetime import datetime
from transformers import pipeline
import tweepy

# ---------------------- INITIALISATION ---------------------- #

def load_twitter_api():
    bearer_token = st.secrets["BEARER_TOKEN"]
    return bearer_token

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
class MyStreamListener(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        if tweet.lang != "fr" or hasattr(tweet, "referenced_tweets"):
            return

        sentiment, score = analyze_sentiment(tweet.text)
        new_tweet = {
            "Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Utilisateur": tweet.author_id,
            "Texte": tweet.text,
            "Sentiment": sentiment,
            "Score": score
        }
        
        st.session_state.tweets = pd.concat(
            [pd.DataFrame([new_tweet]), st.session_state.tweets],
            ignore_index=True
        )
        save_tweets_to_csv()

    def on_errors(self, errors):
        st.error(f"Erreur rencontrée : {errors}")

def collect_tweets(bearer_token, keywords):
    stream = MyStreamListener(bearer_token)
    # Supprimer d'anciennes règles si elles existent
    existing_rules = stream.get_rules().data
    if existing_rules:
        rule_ids = [rule.id for rule in existing_rules]
        stream.delete_rules(rule_ids)
    
    # Ajouter de nouvelles règles
    stream.add_rules(tweepy.StreamRule(value=" OR ".join(keywords)))
    
    # Démarrer le stream dans un thread séparé
    threading.Thread(target=stream.filter, kwargs={'tweet_fields': ['lang', 'author_id'], 'threaded': True}).start()
    return stream

# ---------------------- INTERFACE STREAMLIT ---------------------- #

st.title("🐦 Agent Twitter AI - Dashboard")

keywords_input = st.text_input("🔎 Mots-clés à suivre (séparés par des virgules):", "cryptomonnaie, blockchain, web3, politique, technologies")
keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Démarrer l'agent"):
        if not st.session_state.agent_running:
            bearer_token = load_twitter_api()
            st.session_state.stream = collect_tweets(bearer_token, keywords)
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
