import sqlite3
import bcrypt
from pathlib import Path
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Database File Path ---
DB_PATH = Path("users.db")

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

def get_all_users():
    """
    Retrieves all users from the database.
    Returns a list of user records.
    """
    try:
        conn = get_connection()
        if conn is None:
            return []
        c = conn.cursor()
        c.execute("SELECT * FROM users")
        users = c.fetchall()
        conn.close()
        logger.info("All users retrieved successfully.")
        return users
    except Exception as e:
        logger.error(f"Error fetching all users: {e}")
        return []

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
