import streamlit as st
import pandas as pd
import praw
from mastodon import Mastodon
import sqlite3
import requests
import random
import os
import time
from transformers import pipeline
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# ---------------------- INITIALISATION ---------------------- #

# Chargement des clés API pour Reddit et Mastodon
def load_reddit_api():
    return praw.Reddit(
        client_id=st.secrets.get("REDDIT_CLIENT_ID"),
        client_secret=st.secrets.get("REDDIT_CLIENT_SECRET"),
        user_agent=st.secrets.get("REDDIT_USER_AGENT")
    )

def load_mastodon_api():
    return Mastodon(
        access_token=st.secrets.get("MASTODON_ACCESS_TOKEN"),
        api_base_url=st.secrets.get("MASTODON_API_BASE_URL")
    )

# Chargement du pipeline d'analyse de sentiment adapté aux cryptos & Web3
sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-roberta-base-sentiment", device=-1)

# Connexion à SQLite
conn = sqlite3.connect("data.db", check_same_thread=False)
cursor = conn.cursor()

# Création de la table si elle n'existe pas
cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plateforme TEXT,
        date TEXT,
        utilisateur TEXT,
        texte TEXT,
        sentiment TEXT,
        score REAL
    )
""")
conn.commit()

if "autonomy_enabled" not in st.session_state:
    st.session_state.autonomy_enabled = False

# ---------------------- FONCTIONS UTILES ---------------------- #

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Supprime les balises HTML
    text = re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9 .,!?\n]', '', text)  # Supprime les caractères spéciaux
    return text.strip()

def analyze_sentiment(text):
    if not text or text.isspace():
        return "Neutre", 50.0  # Retourne un score neutre si le texte est vide
    cleaned_text = clean_text(text)[:512]  # Nettoyage et limitation à 512 caractères
    try:
        result = sentiment_analyzer(cleaned_text)[0]  # Passage au modèle NLP
        label = result['label']  # Récupération du sentiment
        score = round(result['score'] * 100, 2)  # Score en pourcentage

        # Adaptation des labels spécifiques à la crypto & Web3
        label_map = {"LABEL_0": "Négatif", "LABEL_1": "Neutre", "LABEL_2": "Positif"}
        sentiment = label_map.get(label, "Neutre")

        return sentiment, score
    except Exception as e:
        st.warning(f"Erreur d'analyse du texte : {e}")
        return "Erreur", 0.0

def store_data(plateforme, date, utilisateur, texte, sentiment, score):
    cursor.execute("INSERT INTO posts (plateforme, date, utilisateur, texte, sentiment, score) VALUES (?, ?, ?, ?, ?, ?)",
                   (plateforme, date, utilisateur, texte, sentiment, score))
    conn.commit()

def fetch_data():
    return pd.read_sql("SELECT * FROM posts ORDER BY date DESC LIMIT 500", conn)

# ---------------------- COLLECTE AUTOMATISÉE ---------------------- #

def collect_reddit_posts(reddit, subreddit_name, limit=50):
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            text = f"{post.title} {(post.selftext or '')}"
            sentiment, score = analyze_sentiment(text)
            store_data("Reddit", datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                       post.author.name if post.author else "Anonyme", clean_text(text), sentiment, score)
    except Exception as e:
        st.warning(f"⚠️ Problème avec l'API Reddit : {e}")

def collect_mastodon_toots(mastodon, hashtag, limit=50):
    try:
        toots = mastodon.timeline_hashtag(hashtag, limit=limit)
        for toot in toots:
            content = clean_text(toot['content'])
            sentiment, score = analyze_sentiment(content)
            store_data("Mastodon", toot['created_at'], toot['account']['username'], content, sentiment, score)
    except Exception as e:
        st.warning(f"⚠️ Problème avec l'API Mastodon : {e}")

def autonomous_agent():
    request_delay = 1800  # 30 minutes
    while st.session_state.autonomy_enabled:
        st.info("🤖 L'agent collecte et analyse les tendances...")
        reddit = load_reddit_api()
        mastodon = load_mastodon_api()
        if reddit:
            collect_reddit_posts(reddit, "cryptocurrency", 50)
        if mastodon:
            collect_mastodon_toots(mastodon, "blockchain", 50)
        st.success("✅ Cycle de collecte terminé.")
        time.sleep(request_delay)

# ---------------------- SUIVI DE L'APPRENTISSAGE ---------------------- #

data = fetch_data()

st.subheader("📊 Évolution des tendances et apprentissage")
if not data.empty:
    st.write("### Évolution des sentiments sur 7 jours")
    data['date'] = pd.to_datetime(data['date'])
    past_week = datetime.now() - timedelta(days=7)
    recent_data = data[data['date'] >= past_week]
    sentiment_counts = recent_data.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='line', ax=ax)
    ax.set_ylabel("Nombre de posts")
    ax.set_xlabel("Date")
    st.pyplot(fig)
else:
    st.info("Aucune donnée récente à analyser.")

# ---------------------- INTERFACE STREAMLIT ---------------------- #

st.title("🌍 Agent AI Autonome - Suivi de l'apprentissage Crypto/Web3")

if st.button("🚀 Activer l'autonomie" if not st.session_state.autonomy_enabled else "⏹️ Désactiver l'autonomie"):
    st.session_state.autonomy_enabled = not st.session_state.autonomy_enabled
    if st.session_state.autonomy_enabled:
        st.success("✅ Autonomie activée ! Collecte et analyse en cours...")
        autonomous_agent()
    else:
        st.warning("🛑 Autonomie désactivée.")

st.markdown("---")

st.subheader("📊 Données collectées et analyse des tendances")
if not data.empty:
    st.dataframe(data.head(20))
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Télécharger les données en CSV", data=csv, file_name="data.csv", mime="text/csv")
else:
    st.info("Aucune donnée collectée pour le moment.")
