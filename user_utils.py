# user_utils.py

import sqlite3
import bcrypt
from pathlib import Path
import logging
import os
import json

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database File Path ---
DB_PATH = Path("users.db")

def get_connection():
    """
    Establishes a connection to the SQLite database.
    Creates the users table if it doesn't exist with the correct schema.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Create users table if it doesn't exist
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()
        logger.info("Connected to the database and ensured users table exists.")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return None

def create_user(username, name, email, password):
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
        c.execute("INSERT INTO users (username, name, email, password) VALUES (?, ?, ?, ?)",
                  (username, name, email, hashed_password))
        conn.commit()
        conn.close()
        logger.info(f"User '{username}' created successfully.")
        return True
    except sqlite3.IntegrityError as ie:
        if 'UNIQUE constraint failed: users.username' in str(ie):
            logger.error(f"Username '{username}' already exists.")
        elif 'UNIQUE constraint failed: users.email' in str(ie):
            logger.error(f"Email '{email}' is already registered.")
        else:
            logger.error(f"Integrity Error: {ie}")
        return False
    except Exception as e:
        logger.error(f"Error creating user '{username}': {e}")
        return False

def get_user(username):
    """
    Retrieves a user from the database by username.
    Returns the user record if found, else None.
    """
    try:
        conn = get_connection()
        if conn is None:
            return None
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if user:
            logger.info(f"User '{username}' retrieved successfully.")
        else:
            logger.info(f"User '{username}' not found.")
        return user
    except Exception as e:
        logger.error(f"Error fetching user '{username}': {e}")
        return None

def verify_password(username, password):
    """
    Verifies a user's password.
    Returns True if the password is correct, else False.
    """
    try:
        user = get_user(username)
        if user:
            stored_password = user[4]  # Assuming password is the 5th column
            is_correct = bcrypt.checkpw(password.encode('utf-8'), stored_password)
            if is_correct:
                logger.info(f"Password for user '{username}' verified successfully.")
            else:
                logger.info(f"Password verification failed for user '{username}'.")
            return is_correct
        else:
            logger.info(f"User '{username}' does not exist for password verification.")
            return False
    except Exception as e:
        logger.error(f"Error verifying password for user '{username}': {e}")
        return False

def reset_password(username, new_password):
    """
    Resets the password for a given username.
    Returns True if successful, else False.
    """
    try:
        conn = get_connection()
        if conn is None:
            return False
        c = conn.cursor()
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        c.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_password, username))
        conn.commit()
        if c.rowcount == 0:
            logger.error(f"User '{username}' not found for password reset.")
            conn.close()
            return False
        conn.close()
        logger.info(f"Password reset successfully for user '{username}'.")
        return True
    except Exception as e:
        logger.error(f"Error resetting password for user '{username}': {e}")
        return False

def load_default_bias_terms():
    """
    Returns the default list of bias terms.
    """
    bias_terms = [
        'always', 'never', 'obviously', 'clearly', 'undoubtedly', 'unquestionably',
        'everyone knows', 'no one believes', 'definitely', 'certainly', 'extremely',
        'inconceivable', 'must', 'prove', 'disprove', 'true', 'false',
        'alarming', 'allegations', 'unfit', 'aggressive', 'alleged',
        'apparently', 'arguably', 'claims', 'controversial', 'disputed',
        'insists', 'questionable', 'reportedly', 'rumored', 'suggests',
        'supposedly', 'unconfirmed', 'suspected', 'reckless', 'radical',
        'extremist', 'biased', 'manipulative', 'deceptive', 'unbelievable',
        'incredible', 'shocking', 'outrageous', 'bizarre', 'absurd',
        'ridiculous', 'disgraceful', 'disgusting', 'horrible', 'terrible',
        'unacceptable', 'unfair', 'scandalous', 'suspicious', 'illegal',
        'illegitimate', 'immoral', 'corrupt', 'criminal', 'dangerous',
        'threatening', 'harmful', 'menacing', 'disturbing', 'distressing',
        'troubling', 'fearful', 'afraid', 'panic', 'terror', 'catastrophe',
        'disaster', 'chaos', 'crisis', 'collapse', 'failure', 'ruin',
        'devastation', 'suffering', 'misery', 'pain', 'dreadful', 'awful',
        'nasty', 'vile', 'vicious', 'brutal', 'violent', 'greedy',
        'selfish', 'arrogant', 'ignorant', 'stupid', 'unwise', 'illogical',
        'unreasonable', 'delusional', 'paranoid', 'obsessed', 'fanatical',
        'zealous', 'militant', 'dictator', 'regime'
    ]
    # Remove duplicates and convert to lowercase
    bias_terms = list(set([term.lower() for term in bias_terms]))
    return bias_terms

def save_analysis_to_history(data):
    """
    Saves analysis data to a JSON file specific to the user's email.
    """
    email = data.get('email', 'guest')
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
    """
    Loads analysis history for a given user.
    """
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
