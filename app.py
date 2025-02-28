import streamlit as st
import pandas as pd
import praw
from mastodon import Mastodon
import requests
import random
import os
import time
from transformers import pipeline
from datetime import datetime
import re
from bs4 import BeautifulSoup

# ---------------------- INITIALISATION ---------------------- #

# Chargement des clés API pour Reddit et Mastodon
def load_reddit_api():
    reddit = praw.Reddit(
        client_id=st.secrets.get("REDDIT_CLIENT_ID"),
        client_secret=st.secrets.get("REDDIT_CLIENT_SECRET"),
        user_agent=st.secrets.get("REDDIT_USER_AGENT")
    )
    return reddit


def load_mastodon_api():
    mastodon = Mastodon(
        access_token=st.secrets.get("MASTODON_ACCESS_TOKEN"),
        api_base_url=st.secrets.get("MASTODON_API_BASE_URL")
    )
    return mastodon

# Chargement du pipeline d'analyse de sentiment avec un modèle léger
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

if "combined_data" not in st.session_state:
    st.session_state.combined_data = pd.DataFrame(columns=["Plateforme", "Date", "Utilisateur", "Texte", "Sentiment", "Score"])
if "autonomy_enabled" not in st.session_state:
    st.session_state.autonomy_enabled = False

# ---------------------- FONCTIONS UTILES ---------------------- #

def clean_text(text):
    """Nettoie le texte en supprimant le HTML, les balises et les caractères spéciaux."""
    text = BeautifulSoup(text, "html.parser").get_text()  # Supprime les balises HTML
    text = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9 .,!?\n]', '', text)  # Supprime les caractères spéciaux
    return text.strip()

def analyze_sentiment(text):
    if not text or text.isspace():
        return "Neutral", 0.0
    cleaned_text = clean_text(text)[:512]  # Limite de 512 caractères
    try:
        result = sentiment_analyzer(cleaned_text)[0]
        return result['label'], round(result['score'] * 100, 2)
    except Exception as e:
        st.warning(f"Erreur d'analyse du texte : {e}")
        return "Error", 0.0

def save_combined_data():
    st.session_state.combined_data.to_csv("combined_data.csv", index=False)

def load_combined_data():
    if os.path.exists("combined_data.csv"):
        st.session_state.combined_data = pd.read_csv("combined_data.csv")

load_combined_data()

# ---------------------- COLLECTE AUTOMATISÉE ---------------------- #

def collect_reddit_posts(reddit, subreddit_name, limit=10):
    posts = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            text = f"{post.title} {(post.selftext or '')}"
            sentiment, score = analyze_sentiment(text)
            posts.append({
                "Plateforme": "Reddit",
                "Date": datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "Utilisateur": post.author.name if post.author else "Anonyme",
                "Texte": clean_text(text),
                "Sentiment": sentiment,
                "Score": score
            })
    except Exception as e:
        st.error(f"🚫 Erreur lors de la collecte Reddit : {e}")
    return posts

def collect_mastodon_toots(mastodon, hashtag, limit=10):
    posts = []
    try:
        toots = mastodon.timeline_hashtag(hashtag, limit=limit)
        for toot in toots:
            content = clean_text(toot['content'])
            sentiment, score = analyze_sentiment(content)
            posts.append({
                "Plateforme": "Mastodon",
                "Date": toot['created_at'],
                "Utilisateur": toot['account']['username'],
                "Texte": content,
                "Sentiment": sentiment,
                "Score": score
            })
    except Exception as e:
        st.error(f"🚫 Erreur lors de la collecte Mastodon : {e}")
    return posts

def autonomous_agent():
    while st.session_state.autonomy_enabled:
        st.info("🤖 L'agent collecte et analyse les tendances...")
        reddit = load_reddit_api()
        mastodon = load_mastodon_api()
        if reddit:
            reddit_posts = collect_reddit_posts(reddit, "cryptocurrency", 10)
            st.session_state.combined_data = pd.concat([st.session_state.combined_data, pd.DataFrame(reddit_posts)], ignore_index=True)
        if mastodon:
            mastodon_posts = collect_mastodon_toots(mastodon, "blockchain", 10)
            st.session_state.combined_data = pd.concat([st.session_state.combined_data, pd.DataFrame(mastodon_posts)], ignore_index=True)
        save_combined_data()
        st.success("✅ Cycle de collecte terminé.")
        time.sleep(1800)  # Attente de 30 minutes

# ---------------------- INTERFACE STREAMLIT ---------------------- #

st.title("🌍 Agent AI Autonome - Reddit | Mastodon")

if st.button("🚀 Activer l'autonomie" if not st.session_state.autonomy_enabled else "⏹️ Désactiver l'autonomie"):
    st.session_state.autonomy_enabled = not st.session_state.autonomy_enabled
    if st.session_state.autonomy_enabled:
        st.success("✅ Autonomie activée ! Collecte et analyse en cours...")
        autonomous_agent()
    else:
        st.warning("🛑 Autonomie désactivée.")

st.markdown("---")

st.subheader("📊 Visualisation des données collectées")
if not st.session_state.combined_data.empty:
    st.dataframe(st.session_state.combined_data.head(20))
else:
    st.info("Aucune donnée collectée pour le moment.")
