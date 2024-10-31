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
import sqlite3
import bcrypt
import subprocess
import sys

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Function to Install SpaCy Model ---
def install_spacy_model(model_name):
    try:
        spacy.load(model_name)
    except OSError:
        logger.info(f"Downloading SpaCy model: {model_name}")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])

# Specify the SpaCy model you want to use
SPACY_MODEL = "en_core_web_sm"

# Ensure the model is installed
install_spacy_model(SPACY_MODEL)

# --- Database Setup ---
DB_PATH = "users.db"

def get_connection():
    """
    Establishes a connection to the SQLite database.
    Creates the users table if it doesn't exist.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Create users table if it doesn't exist
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()
        logger.info("Connected to the database and ensured users table exists.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return None

def create_user(email, name, password):
    """
    Creates a new user with the provided details.
    Passwords are hashed before storing.
    Returns True if successful, else False.
    """
    try:
        conn = get_connection()
        if conn is None:
            return False
        c = conn.cursor()
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        # Insert the new user
        c.execute("INSERT INTO users (email, name, password) VALUES (?, ?, ?)",
                  (email, name, hashed_password))
        conn.commit()
        conn.close()
        logger.info(f"User '{email}' created successfully.")
        return True
    except sqlite3.IntegrityError as ie:
        if 'UNIQUE constraint failed: users.email' in str(ie):
            logger.error(f"Email '{email}' is already registered.")
        else:
            logger.error(f"Integrity Error: {ie}")
        return False
    except Exception as e:
        logger.error(f"Error creating user '{email}': {e}")
        return False

def get_user(email):
    """
    Retrieves a user from the database by email.
    Returns the user record if found, else None.
    """
    try:
        conn = get_connection()
        if conn is None:
            return None
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()
        if user:
            logger.info(f"User with email '{email}' retrieved successfully.")
        else:
            logger.info(f"User with email '{email}' not found.")
        return user
    except Exception as e:
        logger.error(f"Error fetching user '{email}': {e}")
        return None

def verify_password(email, password):
    """
    Verifies a user's password.
    Returns True if the password is correct, else False.
    """
    try:
        user = get_user(email)
        if user:
            stored_password = user[3]  # Assuming password is the 4th column
            is_correct = bcrypt.checkpw(password.encode('utf-8'), stored_password)
            if is_correct:
                logger.info(f"Password for user '{email}' verified successfully.")
            else:
                logger.info(f"Password verification failed for user '{email}'.")
            return is_correct
        else:
            logger.info(f"User '{email}' does not exist for password verification.")
            return False
    except Exception as e:
        logger.error(f"Error verifying password for user '{email}': {e}")
        return False

def reset_password(email, new_password):
    """
    Resets the password for a given email.
    Returns True if successful, else False.
    """
    try:
        conn = get_connection()
        if conn is None:
            return False
        c = conn.cursor()
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        c.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, email))
        conn.commit()
        if c.rowcount == 0:
            logger.error(f"User '{email}' not found for password reset.")
            conn.close()
            return False
        conn.close()
        logger.info(f"Password reset successfully for user '{email}'.")
        return True
    except Exception as e:
        logger.error(f"Error resetting password for user '{email}': {e}")
        return False

# --- Initialize Models ---
@st.cache_resource
def initialize_models():
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
    nlp = spacy.load(SPACY_MODEL)

    models = {
        'sentiment_pipeline': sentiment_pipeline_model,
        'propaganda_pipeline': propaganda_pipeline_model,
        'nlp': nlp
    }
    return models

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

def load_default_bias_terms():
    bias_terms = [
        'alarming',
        'allegations',
        'unfit',
        'aggressive',
        'alleged',
        'apparently',
        'arguably',
        'claims',
        'controversial',
        'disputed',
        'insists',
        'questionable',
        'reportedly',
        'rumored',
        'suggests',
        'supposedly',
        'unconfirmed',
        'suspected',
        'reckless',
        'radical',
        'extremist',
        'biased',
        'manipulative',
        'deceptive',
        'unbelievable',
        'incredible',
        'shocking',
        'outrageous',
        'bizarre',
        'absurd',
        'ridiculous',
        'disgraceful',
        'disgusting',
        'horrible',
        'terrible',
        'unacceptable',
        'unfair',
        'scandalous',
        'suspicious',
        'illegal',
        'illegitimate',
        'immoral',
        'corrupt',
        'criminal',
        'dangerous',
        'threatening',
        'harmful',
        'menacing',
        'disturbing',
        'distressing',
        'troubling',
        'fearful',
        'afraid',
        'panic',
        'terror',
        'catastrophe',
        'disaster',
        'chaos',
        'crisis',
        'collapse',
        'failure',
        'ruin',
        'devastation',
        'suffering',
        'misery',
        'pain',
        'dreadful',
        'awful',
        'nasty',
        'vile',
        'vicious',
        'brutal',
        'violent',
        'greedy',
        'selfish',
        'arrogant',
        'ignorant',
        'stupid',
        'unwise',
        'illogical',
        'unreasonable',
        'delusional',
        'paranoid',
        'obsessed',
        'fanatical',
        'zealous',
        'militant',
        'dictator',
        'regime',
        # Remove duplicates if any
    ]
    # Remove duplicates and convert to lowercase
    bias_terms = list(set([term.lower() for term in bias_terms]))
    return bias_terms

def save_analysis_to_history(data):
    email = st.session_state.get('email', 'guest')
    history_file = f"{email}_history.json"
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        history.append(data)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info(f"Analysis saved to {history_file}.")
    except Exception as e:
        logger.error(f"Error saving analysis to history: {e}")

def load_user_history(email):
    history_file = f"{email}_history.json"
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            return history
        else:
            return []
    except Exception as e:
        logger.error(f"Error loading user history: {e}")
        return []

# --- User Management Functions ---

def register_user():
    st.title("Register")
    st.write("Create a new account to access personalized features.")

    with st.form("registration_form"):
        email = st.text_input("Your Email", key="register_email")
        name = st.text_input("Your Name", key="register_name")
        password = st.text_input("Choose a Password", type='password', key="register_password")
        password_confirm = st.text_input("Confirm Password", type='password', key="register_password_confirm")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not email or not name or not password or not password_confirm:
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
            success = create_user(email, name, password)
            if success:
                st.success("Registration successful. You can now log in.")
                logger.info(f"New user registered: {email}")
            else:
                st.error("Registration failed. Email may already be in use.")
                logger.error(f"Registration failed for user: {email}")

def login_user():
    st.title("Login")
    st.write("Access your account to view history and customize settings.")

    with st.form("login_form"):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type='password', key="login_password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if not email or not password:
                st.error("Please enter both email and password.")
                return
            if verify_password(email, password):
                st.session_state['logged_in'] = True
                st.session_state['email'] = email
                st.session_state['bias_terms'] = load_default_bias_terms()
                st.success("Logged in successfully.")
                logger.info(f"User '{email}' logged in successfully.")
            else:
                st.error("Invalid email or password.")
                logger.warning(f"Failed login attempt for email: '{email}'.")

    st.markdown("---")
    st.write("Forgot your password?")
    if st.button("Reset Password"):
        reset_password_flow()

def reset_password_flow():
    st.title("Reset Password")
    st.write("Enter your email and new password.")

    with st.form("reset_password_form"):
        email = st.text_input("Email", key="reset_email")
        new_password = st.text_input("New Password", type='password', key="new_password")
        new_password_confirm = st.text_input("Confirm New Password", type='password', key="new_password_confirm")
        reset_button = st.form_submit_button("Reset Password")

        if reset_button:
            if not email or not new_password or not new_password_confirm:
                st.error("Please fill out all fields.")
                return
            if not is_valid_email(email):
                st.error("Please enter a valid email address.")
                return
            if new_password != new_password_confirm:
                st.error("Passwords do not match.")
                return
            if not is_strong_password(new_password):
                st.error("Password must be at least 8 characters long, include at least one uppercase letter, one digit, and one special character.")
                return
            success = reset_password(email, new_password)
            if success:
                st.success("Password reset successful. You can now log in.")
                logger.info(f"User '{email}' reset their password.")
            else:
                st.error("Password reset failed. Email may not exist.")
                logger.error(f"Password reset failed for user: {email}")

def logout_user():
    logger.info(f"User '{st.session_state['email']}' logged out.")
    st.session_state['logged_in'] = False
    st.session_state['email'] = ''
    st.session_state['bias_terms'] = []
    st.sidebar.success("Logged out successfully.")

# --- Analysis Functions ---

def perform_analysis(text, title="Article"):
    # [Function body remains the same]
    # No changes required here

# --- Display Functions ---

def display_results(data, is_nested=False):
    # [Function body remains the same]
    # No changes required here

# --- Single Article Analysis Function ---

def single_article_analysis():
    # [Function body remains the same]
    # No changes required here

# --- Comparative Analysis Function ---

def comparative_analysis():
    # [Function body remains the same]
    # No changes required here

def display_comparative_results(analyses):
    # [Function body remains the same]
    # No changes required here

# --- Settings Page Function ---

def settings_page():
    # [Function body remains the same]
    # No changes required here

# --- Help Page Function ---

def help_feature():
    # [Function body remains the same]
    # No changes required here

# --- History Page Function ---

def display_history():
    # [Function body remains the same]
    # No changes required here

# --- Main Function ---

def main():
    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
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
            st.sidebar.info(f"Already logged in as **{st.session_state['email']}**.")
        else:
            login_user()
    elif page == "Register":
        if st.session_state['logged_in']:
            st.sidebar.info(f"Already registered as **{st.session_state['email']}**.")
        else:
            register_user()
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
        st.sidebar.markdown(f"**Logged in as:** {st.session_state['email']}")
        if st.sidebar.button("Logout"):
            logout_user()

if __name__ == "__main__":
    main()
