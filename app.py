import streamlit as st
import pandas as pd
import os
from datetime import datetime
from transformers import pipeline
import tweepy

# ---------------------- INITIALISATION ---------------------- #

def load_twitter_api():
    api_key = st.secrets.get("6HeKGQ0Ja5j8a7m0whFa2RmtS")
    api_secret = st.secrets.get("jm8VamfS3ysMr8JD5zBQTeWWd183deWKYyGvQmVdlmlPOLKvdn")
    access_token = st.secrets.get("1894174463119495168-uVebuhJ3sQOac4kovU6Xnl9HJrmD8y")
    access_token_secret = st.secrets.get("9Q0X6HSPA3J7ltpi46R8YogqqiUMHQWgq4cAAmEe3DHcL")

    if not all([api_key, api_secret, access_token, access_token_secret]):
        st.error("❌ Clés API manquantes. Vérifiez vos secrets Streamlit.")
        return None

    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

sentiment_analyzer = pipeline("sentiment-analysis")

# Initialisation des états Streamlit
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

# ---------------------- COLLECTE DE TWEETS ---------------------- #

def collect_recent_tweets(api, keywords, max_results=20):
    query = " OR ".join([k.strip() for k in keywords if k.strip()]) + " -is:retweet lang:fr"
    
    try:
        tweets = api.search_tweets(q=query, count=max_results, tweet_mode="extended", lang="fr")
        
        if not tweets:
            st.info("🔍 Aucun tweet trouvé pour les mots-clés fournis.")
            return

        new_tweets = []
        for tweet in tweets:
            sentiment, score = analyze_sentiment(tweet.full_text)
            new_tweets.append({
                "Date": tweet.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                "Utilisateur": tweet.user.screen_name,
                "Texte": tweet.full_text,
                "Sentiment": sentiment,
                "Score": score
            })

        st.session_state.tweets = pd.concat([
            pd.DataFrame(new_tweets),
            st.session_state.tweets
        ], ignore_index=True).drop_duplicates(subset=["Texte"])
        
        save_tweets_to_csv()
        st.success(f"✅ {len(new_tweets)} tweets collectés avec succès.")

    except tweepy.TweepyException as e:
        st.error(f"🚫 Erreur lors de la collecte des tweets : {e}")

# ---------------------- INTERFACE STREAMLIT ---------------------- #

st.title("🐦 Agent Twitter AI - Dashboard (Version Gratuite)")

keywords_input = st.text_input("🔎 Mots-clés à rechercher (séparés par des virgules):", "cryptomonnaie, blockchain, web3, politique, technologies")
keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

max_results = st.slider("Nombre de tweets à collecter par recherche :", min_value=5, max_value=50, value=20, step=5)

if st.button("📥 Collecter les tweets"):
    api = load_twitter_api()
    if api:
        collect_recent_tweets(api, keywords, max_results)

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
