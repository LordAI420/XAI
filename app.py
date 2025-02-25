import streamlit as st
import pandas as pd
import os
import requests
import random
from datetime import datetime
from transformers import pipeline
import tweepy
from streamlit_autorefresh import st_autorefresh

# ---------------------- INITIALISATION ---------------------- #

def load_twitter_api():
    api_key = st.secrets.get("API_KEY")
    api_secret = st.secrets.get("API_SECRET")
    access_token = st.secrets.get("ACCESS_TOKEN")
    access_token_secret = st.secrets.get("ACCESS_TOKEN_SECRET")

    if not all([api_key, api_secret, access_token, access_token_secret]):
        st.error("❌ Clés API manquantes. Vérifiez vos secrets Streamlit.")
        return None

    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

def load_bearer_token():
    bearer_token = st.secrets.get("BEARER_TOKEN")
    if not bearer_token:
        st.error("❌ Bearer Token manquant. Vérifiez vos secrets Streamlit.")
    return bearer_token

sentiment_analyzer = pipeline("sentiment-analysis")

if "tweets" not in st.session_state:
    st.session_state.tweets = pd.DataFrame(columns=["Date", "Utilisateur", "Texte", "Sentiment", "Score"])

if "autonomy_enabled" not in st.session_state:
    st.session_state.autonomy_enabled = False

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

def collect_recent_tweets(bearer_token, keywords, max_results=10):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    query = " OR ".join([k.strip() for k in keywords if k.strip()]) + " -is:retweet lang:fr"

    url = (
        f"https://api.twitter.com/2/tweets/search/recent?"
        f"query={query}&max_results={max_results}&tweet.fields=author_id,created_at"
    )

    response = requests.get(url, headers=headers)

    if response.status_code == 429:
        st.warning("⚠️ Limite d'utilisation de l'API atteinte. Veuillez réessayer plus tard.")
        return []

    if response.status_code != 200:
        st.error(f"🚫 Erreur lors de la collecte des tweets : {response.status_code} - {response.json()}")
        return []

    data = response.json()
    tweets = data.get("data", [])

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
    return new_tweets

# ---------------------- GÉNÉRATION ET PUBLICATION DE TWEETS ---------------------- #

def generate_tweet_from_trends(tweets):
    if tweets.empty:
        return "Rien de nouveau pour le moment. Restez connectés !"

    top_sentiment = tweets['Sentiment'].mode()[0]
    frequent_words = pd.Series(' '.join(tweets['Texte']).lower().split()).value_counts().head(5).index.tolist()
    trend = random.choice(frequent_words) if frequent_words else "innovation"

    tweet_templates = [
        f"La discussion autour de #{trend} est intense aujourd'hui. Que pensez-vous de cette tendance ? 🤔",
        f"Les conversations sur #{trend} montrent un sentiment {top_sentiment.lower()}. Partagez votre avis !",
        f"#{trend} est au cœur des débats actuellement. Quelle est votre opinion ? 💭"
    ]

    return random.choice(tweet_templates)

def publish_tweet(api, tweet_text):
    try:
        api.update_status(tweet_text)
        st.success(f"✅ Tweet publié avec succès : {tweet_text}")
    except Exception as e:
        st.error(f"🚫 Erreur lors de la publication du tweet : {e}")

# ---------------------- LOGIQUE D'AUTONOMIE ---------------------- #

def autonomous_agent(api, bearer_token, keywords, max_results=5):
    new_tweets = collect_recent_tweets(bearer_token, keywords, max_results)
    if new_tweets:
        generated_tweet = generate_tweet_from_trends(st.session_state.tweets)
        publish_tweet(api, generated_tweet)

# ---------------------- INTERFACE STREAMLIT ---------------------- #

st.title("🐦 Agent Twitter AI Autonome - Dashboard")

keywords_input = st.text_input("🔎 Mots-clés à suivre (séparés par des virgules):", "cryptomonnaie, blockchain, web3, politique, technologies")
keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

max_results = st.slider("Nombre de tweets à collecter par recherche :", min_value=1, max_value=10, value=5, step=1)

# Activation/Désactivation de l'autonomie
if st.button("🚀 Activer l'autonomie" if not st.session_state.autonomy_enabled else "⏹️ Désactiver l'autonomie"):
    st.session_state.autonomy_enabled = not st.session_state.autonomy_enabled
    st.success("✅ Autonomie activée." if st.session_state.autonomy_enabled else "🛑 Autonomie désactivée.")

# Si autonomie activée, exécution automatique toutes les 60 minutes (pour éviter la limite d'API)
if st.session_state.autonomy_enabled:
    st.info("🤖 L'agent est en mode autonome. Collecte et publication automatiques toutes les 60 minutes.")
    api = load_twitter_api()
    bearer_token = load_bearer_token()
    if api and bearer_token:
        autonomous_agent(api, bearer_token, keywords, max_results)
    st_autorefresh(interval=3600000, key="autorefresh")  # Refresh toutes les 60 minutes

# Boutons manuels pour collecte et génération de tweets
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Collecter les tweets"):
        bearer_token = load_bearer_token()
        if bearer_token:
            collected_tweets = collect_recent_tweets(bearer_token, keywords, max_results)
            st.success(f"✅ {len(collected_tweets)} tweets collectés.")

with col2:
    if st.button("✍️ Générer et publier un tweet basé sur les tendances"):
        api = load_twitter_api()
        if api and not st.session_state.tweets.empty:
            generated_tweet = generate_tweet_from_trends(st.session_state.tweets)
            publish_tweet(api, generated_tweet)
        else:
            st.warning("⚠️ Collectez d'abord des tweets pour générer un message.")

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
