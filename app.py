import streamlit as st
import pandas as pd
import praw
from mastodon import Mastodon
import requests
import random
import os
from transformers import pipeline
from datetime import datetime

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

sentiment_analyzer = pipeline("sentiment-analysis")

if "combined_data" not in st.session_state:
    st.session_state.combined_data = pd.DataFrame(columns=["Plateforme", "Date", "Utilisateur", "Texte", "Sentiment", "Score"])

# ---------------------- COLLECTE DE DONNÉES ---------------------- #

# Reddit - Collecte de posts
def collect_reddit_posts(reddit, subreddit_name, limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        sentiment, score = analyze_sentiment(post.title + " " + (post.selftext or ""))
        posts.append({
            "Plateforme": "Reddit",
            "Date": datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            "Utilisateur": post.author.name if post.author else "Anonyme",
            "Texte": post.title + " " + (post.selftext or ""),
            "Sentiment": sentiment,
            "Score": score
        })
    return posts

# Mastodon - Collecte de toots
def collect_mastodon_toots(mastodon, hashtag, limit=10):
    toots = mastodon.timeline_hashtag(hashtag, limit=limit)
    posts = []
    for toot in toots:
        content = toot['content']
        sentiment, score = analyze_sentiment(content)
        posts.append({
            "Plateforme": "Mastodon",
            "Date": toot['created_at'],
            "Utilisateur": toot['account']['username'],
            "Texte": content,
            "Sentiment": sentiment,
            "Score": score
        })
    return posts

# Jeux de données open source - Import de CSV
def load_open_source_dataset(file):
    df = pd.read_csv(file)
    df['Sentiment'], df['Score'] = zip(*df['Texte'].map(analyze_sentiment))
    df['Plateforme'] = 'Dataset'
    return df[['Plateforme', 'Date', 'Utilisateur', 'Texte', 'Sentiment', 'Score']]

# Analyse de sentiments
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], round(result['score'] * 100, 2)

# Sauvegarde des données
def save_combined_data():
    st.session_state.combined_data.to_csv("combined_data.csv", index=False)

# Chargement des données existantes
def load_combined_data():
    if os.path.exists("combined_data.csv"):
        st.session_state.combined_data = pd.read_csv("combined_data.csv")

load_combined_data()

# ---------------------- GÉNÉRATION DE CONTENU ---------------------- #

def generate_trend_based_post(data):
    if data.empty:
        return "Rien de nouveau pour le moment. Restez connectés !"

    frequent_words = pd.Series(' '.join(data['Texte']).lower().split()).value_counts().head(5).index.tolist()
    trend = random.choice(frequent_words) if frequent_words else "innovation"
    top_sentiment = data['Sentiment'].mode()[0]

    post_templates = [
        f"La discussion sur #{trend} est en plein essor aujourd'hui. Partagez vos pensées !",
        f"Les utilisateurs ressentent principalement un sentiment {top_sentiment.lower()} autour de #{trend}. Qu'en pensez-vous ?",
        f"#{trend} est au cœur des débats. Voici ce qui est dit : {random.choice(data['Texte'].tolist())}"
    ]

    return random.choice(post_templates)

# ---------------------- INTERFACE STREAMLIT ---------------------- #

st.title("🌍 Agent AI Multiplateforme - Reddit | Mastodon | Open Datasets")

col1, col2, col3 = st.columns(3)

# Collecte sur Reddit
with col1:
    st.subheader("📥 Collecte Reddit")
    subreddit_name = st.text_input("Nom du subreddit:", "cryptocurrency")
    reddit_limit = st.slider("Nombre de posts à collecter:", 1, 20, 10)
    if st.button("Collecter depuis Reddit"):
        reddit = load_reddit_api()
        reddit_posts = collect_reddit_posts(reddit, subreddit_name, reddit_limit)
        st.session_state.combined_data = pd.concat([st.session_state.combined_data, pd.DataFrame(reddit_posts)], ignore_index=True)
        save_combined_data()
        st.success(f"✅ {len(reddit_posts)} posts collectés depuis Reddit.")

# Collecte sur Mastodon
with col2:
    st.subheader("🐘 Collecte Mastodon")
    hashtag = st.text_input("Hashtag à suivre (sans #):", "blockchain")
    mastodon_limit = st.slider("Nombre de toots à collecter:", 1, 20, 10)
    if st.button("Collecter depuis Mastodon"):
        mastodon = load_mastodon_api()
        mastodon_posts = collect_mastodon_toots(mastodon, hashtag, mastodon_limit)
        st.session_state.combined_data = pd.concat([st.session_state.combined_data, pd.DataFrame(mastodon_posts)], ignore_index=True)
        save_combined_data()
        st.success(f"✅ {len(mastodon_posts)} toots collectés depuis Mastodon.")

# Import de jeux de données
with col3:
    st.subheader("📂 Importer un dataset")
    uploaded_file = st.file_uploader("Choisir un fichier CSV:")
    if uploaded_file:
        dataset_df = load_open_source_dataset(uploaded_file)
        st.session_state.combined_data = pd.concat([st.session_state.combined_data, dataset_df], ignore_index=True)
        save_combined_data()
        st.success("✅ Données du fichier importées avec succès.")

st.markdown("---")

st.subheader("📊 Visualisation des données combinées")
if not st.session_state.combined_data.empty:
    st.dataframe(st.session_state.combined_data.head(20))
    sentiment_counts = st.session_state.combined_data['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
else:
    st.info("Aucune donnée collectée pour le moment.")

st.markdown("---")

st.subheader("✍️ Génération de contenu basé sur les tendances")
if st.button("Générer un post basé sur les tendances"):
    generated_post = generate_trend_based_post(st.session_state.combined_data)
    st.success(f"📝 Post généré : {generated_post}")

st.markdown("---")

st.subheader("💾 Exporter les données")
if not st.session_state.combined_data.empty:
    csv = st.session_state.combined_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données combinées en CSV",
        data=csv,
        file_name="combined_data.csv",
        mime="text/csv",
    )
else:
    st.info("Collectez ou importez des données avant de pouvoir les télécharger.")
