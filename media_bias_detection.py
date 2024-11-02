# media_bias_detection.py

import streamlit as st
import logging
import datetime
import os
import json
import pandas as pd
from transformers import pipeline
import spacy
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import unicodedata
import ssl
import plotly.express as px
import torch
import sys

# Import user management functions
from user_utils import (
    create_user,
    get_user,
    verify_password,
    reset_password,
    load_default_bias_terms,
    save_analysis_to_history,
    load_user_history
)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Initialize Models ---
@st.cache_resource
def initialize_models():
    try:
        # Initialize Sentiment Analysis Model
        sentiment_pipeline_model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1  # Use CPU
        )
        # Initialize Propaganda Detection Model
        propaganda_pipeline_model = pipeline(
            "text-classification",
            model="IDA-SERICS/PropagandaDetection",
            device=-1  # Use CPU
        )
        # Initialize SpaCy NLP Model
        nlp = spacy.load("en_core_web_sm")  # Directly load the pre-installed model

        models = {
            'sentiment_pipeline': sentiment_pipeline_model,
            'propaganda_pipeline': propaganda_pipeline_model,
            'nlp': nlp
        }
        return models
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        st.error("Failed to initialize NLP models. Please check the logs for more details.")
        return None

# --- Helper Functions ---

def is_strong_password(password):
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True

def is_valid_email(email):
    return re.match(r'^[^@]+@[^@]+\.[^@]+$', email) is not None

def is_valid_url(url):
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])

async def fetch_article_text_async(url):
    if not is_valid_url(url):
        st.error("Invalid URL format.")
        return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, ssl=ssl_context, timeout=10) as response:
                if response.status != 200:
                    st.error(f"HTTP Error: {response.status}")
                    return None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                article_text = ''
                main_content = soup.find('main')
                if main_content:
                    article_text = main_content.get_text(separator=' ', strip=True)
                else:
                    paragraphs = soup.find_all('p')
                    article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
                if not article_text.strip():
                    st.error("No content found at the provided URL.")
                    return None
                return article_text
    except Exception as e:
        st.error(f"Error fetching the article: {e}")
        logger.error(f"Error fetching the article: {e}")
        return None

def fetch_article_text(url):
    try:
        article_text = asyncio.run(fetch_article_text_async(url))
        return article_text
    except Exception as e:
        st.error(f"Error in fetching article text: {e}")
        logger.error(f"Error in fetching article text: {e}")
        return None

def preprocess_text(text):
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Remove unwanted characters and correct encoding issues
    text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Additional cleaning steps can be added here
    return text

def load_custom_bias_terms(user_bias_terms):
    """
    Combines default bias terms with user-defined bias terms.
    """
    default_terms = load_default_bias_terms()
    user_terms = [term.lower() for term in user_bias_terms]
    combined_terms = list(set(default_terms + user_terms))
    return combined_terms

# --- User Management Functions ---

def register_user_ui():
    st.title("Register")
    st.write("Create a new account to access personalized features.")

    with st.form("registration_form"):
        username = st.text_input("Username", key="register_username")
        email = st.text_input("Your Email", key="register_email")
        name = st.text_input("Your Name", key="register_name")
        password = st.text_input("Choose a Password", type='password', key="register_password")
        password_confirm = st.text_input("Confirm Password", type='password', key="register_password_confirm")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not username or not email or not name or not password or not password_confirm:
                st.error("Please fill out all fields.")
                return
            if not is_valid_email(email):
                st.error("Please enter a valid email address.")
                return
            if password != password_confirm:
                st.error("Passwords do not match.")
                return
            if not is_strong_password(password):
                st.error("Password must be at least 8 characters long, include at least one uppercase letter, one digit, and one special character.")
                return
            success = create_user(username, name, email, password)
            if success:
                st.success("Registration successful. You can now log in.")
                logger.info(f"New user registered: {username}")
            else:
                st.error("Registration failed. Username or Email may already be in use.")
                logger.error(f"Registration failed for user: {username}")

def login_user_ui():
    st.title("Login")
    st.write("Access your account to view history and customize settings.")

    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type='password', key="login_password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if not username or not password:
                st.error("Please enter both username and password.")
                return
            if verify_password(username, password):
                user = get_user(username)
                if user:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state['email'] = user['email']  # Corrected access
                    st.session_state['bias_terms'] = load_default_bias_terms()
                    st.success("Logged in successfully.")
                    logger.info(f"User '{username}' logged in successfully.")
                else:
                    st.error("User data not found.")
                    logger.error(f"User data missing for '{username}' despite successful password verification.")
            else:
                st.error("Invalid username or password.")
                logger.warning(f"Failed login attempt for username: '{username}'.")

def reset_password_flow_ui():
    st.title("Reset Password")
    st.write("Enter your username and new password.")

    with st.form("reset_password_form"):
        username = st.text_input("Username", key="reset_username")
        new_password = st.text_input("New Password", type='password', key="new_password")
        new_password_confirm = st.text_input("Confirm New Password", type='password', key="new_password_confirm")
        reset_button = st.form_submit_button("Reset Password")

        if reset_button:
            if not username or not new_password or not new_password_confirm:
                st.error("Please fill out all fields.")
                return
            if new_password != new_password_confirm:
                st.error("Passwords do not match.")
                return
            if not is_strong_password(new_password):
                st.error("Password must be at least 8 characters long, include at least one uppercase letter, one digit, and one special character.")
                return
            success = reset_password(username, new_password)
            if success:
                st.success("Password reset successful. You can now log in.")
                logger.info(f"User '{username}' reset their password.")
            else:
                st.error("Password reset failed. Username may not exist.")
                logger.error(f"Password reset failed for user: {username}")

def logout_user():
    logger.info(f"User '{st.session_state['username']}' logged out.")
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''
    st.session_state['email'] = ''
    st.session_state['bias_terms'] = []
    st.sidebar.success("Logged out successfully.")

# --- Analysis Functions ---

def perform_analysis(text, title="Article"):
    models = initialize_models()
    if models is None:
        st.error("NLP models are not initialized. Cannot perform analysis.")
        return None

    sentiment_pipeline = models['sentiment_pipeline']
    propaganda_pipeline = models['propaganda_pipeline']
    nlp = models['nlp']

    # Preprocess text
    cleaned_text = preprocess_text(text)

    # Sentiment Analysis
    sentiment_result = sentiment_pipeline(cleaned_text[:512])  # Limiting to first 512 tokens

    # Propaganda Detection
    propaganda_result = propaganda_pipeline(cleaned_text[:512])

    # Named Entity Recognition using SpaCy
    doc = nlp(cleaned_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Bias Term Detection
    bias_terms = st.session_state.get('bias_terms', [])
    bias_found = [term for term in bias_terms if term in cleaned_text.lower()]

    analysis = {
        'title': title,
        'sentiment': sentiment_result,
        'propaganda': propaganda_result,
        'entities': entities,
        'bias_terms_found': bias_found,
        'timestamp': datetime.datetime.now().isoformat(),
        'username': st.session_state.get('username', 'guest'),
        'email': st.session_state.get('email', 'guest')
    }

    # Save to history
    save_analysis_to_history(analysis)

    return analysis

# --- Display Functions ---

def display_results(data, is_nested=False):
    st.subheader(f"Analysis Results for: {data['title']}")

    # Sentiment Analysis
    st.markdown("### Sentiment Analysis")
    sentiment = data['sentiment'][0]
    st.write(f"**Label:** {sentiment['label']}")
    st.write(f"**Score:** {sentiment['score']:.2f}")

    # Propaganda Detection
    st.markdown("### Propaganda Detection")
    propaganda = data['propaganda'][0]
    st.write(f"**Label:** {propaganda['label']}")
    st.write(f"**Score:** {propaganda['score']:.2f}")

    # Named Entities
    st.markdown("### Named Entities")
    if data['entities']:
        entities_df = pd.DataFrame(data['entities'], columns=['Entity', 'Type'])
        st.dataframe(entities_df)
    else:
        st.write("No named entities found.")

    # Bias Terms
    st.markdown("### Detected Bias Terms")
    if data['bias_terms_found']:
        bias_terms_str = ', '.join(data['bias_terms_found'])
        st.write(f"**Bias Terms Found:** {bias_terms_str}")
    else:
        st.write("No bias terms detected.")

    # Visualizations
    st.markdown("### Visualizations")

    # Sentiment Pie Chart
    sentiment_labels = [sentiment['label']]
    sentiment_values = [sentiment['score']]
    fig_sentiment = px.pie(names=sentiment_labels, values=sentiment_values, title='Sentiment Distribution')
    st.plotly_chart(fig_sentiment)

    # Propaganda Bar Chart
    propaganda_labels = [propaganda['label']]
    propaganda_values = [propaganda['score']]
    fig_propaganda = px.bar(x=propaganda_labels, y=propaganda_values, title='Propaganda Detection Score')
    st.plotly_chart(fig_propaganda)

def display_comparative_results(analyses):
    st.subheader("Comparative Analysis Results")

    # Sentiment Comparison
    st.markdown("### Sentiment Comparison")
    sentiment_data = {}
    for analysis in analyses:
        label = analysis['sentiment'][0]['label']
        score = analysis['sentiment'][0]['score']
        sentiment_data[analysis['title']] = score
    sentiment_df = pd.DataFrame.from_dict(sentiment_data, orient='index', columns=['Sentiment Score'])
    fig_sentiment = px.bar(sentiment_df, x=sentiment_df.index, y='Sentiment Score',
                           title='Sentiment Comparison', labels={'x': 'Article', 'Sentiment Score': 'Score'})
    st.plotly_chart(fig_sentiment)

    # Propaganda Comparison
    st.markdown("### Propaganda Comparison")
    propaganda_data = {}
    for analysis in analyses:
        label = analysis['propaganda'][0]['label']
        score = analysis['propaganda'][0]['score']
        propaganda_data[analysis['title']] = score
    propaganda_df = pd.DataFrame.from_dict(propaganda_data, orient='index', columns=['Propaganda Score'])
    fig_propaganda = px.bar(propaganda_df, x=propaganda_df.index, y='Propaganda Score',
                            title='Propaganda Detection Comparison', labels={'x': 'Article', 'Propaganda Score': 'Score'})
    st.plotly_chart(fig_propaganda)

    # Bias Terms Comparison
    st.markdown("### Bias Terms Comparison")
    bias_data = {}
    for analysis in analyses:
        bias_count = len(analysis['bias_terms_found'])
        bias_data[analysis['title']] = bias_count
    bias_df = pd.DataFrame.from_dict(bias_data, orient='index', columns=['Bias Terms Found'])
    fig_bias = px.bar(bias_df, x=bias_df.index, y='Bias Terms Found',
                      title='Bias Terms Detection Comparison', labels={'x': 'Article', 'Bias Terms Found': 'Count'})
    st.plotly_chart(fig_bias)

# --- Single Article Analysis Function ---

def single_article_analysis():
    st.title("Single Article Analysis")
    st.write("Analyze the bias and sentiment of a single news article.")

    with st.form("single_article_form"):
        url = st.text_input("Enter the URL of the article:", key="single_article_url")
        uploaded_file = st.file_uploader("Or upload a text file containing the article:", type=["txt"])
        submitted = st.form_submit_button("Analyze")

        if submitted:
            if not url and not uploaded_file:
                st.error("Please provide a URL or upload a text file.")
                return
            if url:
                article_text = fetch_article_text(url)
                title = url
            else:
                try:
                    article_text = uploaded_file.read().decode('utf-8')
                    title = uploaded_file.name
                except Exception as e:
                    st.error("Error reading the uploaded file.")
                    logger.error(f"Error reading uploaded file: {e}")
                    return
            if article_text:
                with st.spinner("Performing analysis..."):
                    analysis = perform_analysis(article_text, title=title)
                if analysis:
                    display_results(analysis)

# --- Comparative Analysis Function ---

def comparative_analysis():
    st.title("Comparative Analysis")
    st.write("Compare the bias and sentiment across multiple news articles.")

    with st.form("comparative_analysis_form"):
        urls = st.text_area("Enter the URLs of the articles (one per line):", key="comparative_urls")
        uploaded_files = st.file_uploader("Or upload multiple text files containing the articles:", type=["txt"], accept_multiple_files=True)
        submitted = st.form_submit_button("Analyze")

        if submitted:
            if not urls and not uploaded_files:
                st.error("Please provide URLs or upload text files.")
                return
            articles = []
            titles = []
            if urls:
                url_list = urls.strip().split('\n')
                for url in url_list:
                    url = url.strip()
                    if url:
                        text = fetch_article_text(url)
                        if text:
                            articles.append(text)
                            titles.append(url)
            if uploaded_files:
                for file in uploaded_files:
                    try:
                        text = file.read().decode('utf-8')
                        articles.append(text)
                        titles.append(file.name)
                    except Exception as e:
                        st.error(f"Error reading file {file.name}.")
                        logger.error(f"Error reading file {file.name}: {e}")
            if articles:
                analyses = []
                with st.spinner("Performing analysis on all articles..."):
                    for text, title in zip(articles, titles):
                        analysis = perform_analysis(text, title=title)
                        if analysis:
                            analyses.append(analysis)
                if analyses:
                    display_comparative_results(analyses)
            else:
                st.error("No valid articles to analyze.")

# --- Settings Page Function ---

def settings_page():
    st.title("Settings")
    st.write("Customize your preferences.")

    with st.form("settings_form"):
        st.markdown("### Customize Bias Terms")
        custom_bias_terms = st.text_area(
            "Add your own bias terms (separated by commas):",
            value="",
            help="Example: biased, manipulative, deceptive"
        )
        submitted = st.form_submit_button("Update Settings")

        if submitted:
            if custom_bias_terms:
                user_terms = [term.strip() for term in custom_bias_terms.split(',') if term.strip()]
                st.session_state['bias_terms'] = load_custom_bias_terms(user_terms)
                st.success("Bias terms updated successfully.")
                logger.info(f"User '{st.session_state['username']}' updated bias terms.")
            else:
                st.session_state['bias_terms'] = load_default_bias_terms()
                st.success("Bias terms reset to default.")
                logger.info(f"User '{st.session_state['username']}' reset bias terms to default.")

# --- Help Page Function ---

def help_feature():
    st.title("Help & Documentation")
    st.write("""
    ## Media Bias Detection Tool

    ### Features
    - **User Authentication**: Register, login, and manage your account securely.
    - **Single Article Analysis**: Analyze sentiment, propaganda, named entities, and bias terms in a single article.
    - **Comparative Analysis**: Compare multiple articles to identify differences in sentiment and bias.
    - **Customizable Bias Terms**: Add your own bias terms to tailor the analysis to your preferences.
    - **History**: View your past analyses for reference.

    ### How to Use
    1. **Register**: Create a new account using the Register page.
    2. **Login**: Access your account using the Login page.
    3. **Single Article Analysis**: Provide a URL or upload a text file to analyze a single article.
    4. **Comparative Analysis**: Provide multiple URLs or upload multiple text files to compare articles.
    5. **Settings**: Customize your bias terms to enhance analysis accuracy.
    6. **History**: Review your past analyses and results.

    ### Support
    If you encounter any issues or have questions, please contact support at [support@example.com](mailto:support@example.com).
    """)

# --- History Page Function ---

def display_history():
    st.title("Analysis History")
    st.write("View your past analyses.")

    email = st.session_state.get('email', 'guest')
    history = load_user_history(email)

    if not history:
        st.info("No analysis history found.")
        return

    for analysis in history:
        st.markdown("---")
        display_results(analysis, is_nested=True)

# --- Main Function ---

def main():
    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = ''
    if 'email' not in st.session_state:
        st.session_state['email'] = ''
    if 'bias_terms' not in st.session_state:
        st.session_state['bias_terms'] = load_default_bias_terms()

    # Sidebar Navigation
    st.sidebar.title("Media Bias Detection Tool")
    st.sidebar.markdown("---")
    if not st.session_state['logged_in']:
        page = st.sidebar.radio(
            "Navigate to",
            ["Login", "Register", "Help"]
        )
    else:
        page = st.sidebar.radio(
            "Navigate to",
            ["Single Article Analysis", "Comparative Analysis", "History", "Settings", "Help"]
        )
    st.sidebar.markdown("---")

    # Page Routing
    if page == "Login":
        if st.session_state['logged_in']:
            st.sidebar.info(f"Already logged in as **{st.session_state['username']}**.")
        else:
            login_user_ui()
    elif page == "Register":
        if st.session_state['logged_in']:
            st.sidebar.info(f"Already registered as **{st.session_state['username']}**.")
        else:
            register_user_ui()
    elif page == "Single Article Analysis":
        if not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
        else:
            single_article_analysis()
    elif page == "Comparative Analysis":
        if not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
        else:
            comparative_analysis()
    elif page == "History":
        if not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
        else:
            display_history()
    elif page == "Settings":
        if not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
        else:
            settings_page()
    elif page == "Help":
        help_feature()

    # Logout Option
    if st.session_state['logged_in'] and page not in ["Login", "Register"]:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            logout_user()

if __name__ == "__main__":
    main()
