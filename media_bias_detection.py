import os
import sys
import json
import logging
import hashlib
import sqlite3
import argparse
from typing import List, Tuple, Dict
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    hamming_loss,
    f1_score,
    precision_recall_fscore_support
)
import joblib
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from bertopic import BERTopic
from collections import defaultdict
import streamlit as st
import swifter  # For efficient dataframe operations

# ----------------------------
# Initialization
# ----------------------------

# Download necessary NLTK data if not already present.
nltk_packages = ['wordnet', 'omw-1.4', 'stopwords']
for pkg in nltk_packages:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

# Initialize spaCy with disabled parser and NER for optimized processing.
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    st.write("Downloading spaCy 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize NER model
try:
    ner_model = spacy.load("en_core_web_sm")
except OSError:
    st.write("Downloading spaCy 'en_core_web_sm' model for NER...")
    from spacy.cli import download
    download("en_core_web_sm")
    ner_model = spacy.load("en_core_web_sm")

# Relevant Entity Labels for NER Enhancement
RELEVANT_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC"}

# ----------------------------
# Configuration
# ----------------------------

DEFAULT_CONFIG = {
    'data_file': 'models/Propaganda_Dataset.json',               # Input JSON data file
    'propaganda_techniques_file': 'models/propaganda_techniques.json',  # Techniques JSON file
    'bias_model_file': 'models/media_bias_model',                # Media bias model path
    'propaganda_model_file': 'models/propaganda_detection_model', # Propaganda model path
    'label_encoder_file': 'models/label_encoder.pkl',            # Label encoder path
    'log_file': 'logs/propaganda_detection.log',                 # Log file path
    'bias_report_file': 'reports/media_bias_report.html',        # Report file for media bias detection
    'propaganda_report_file': 'reports/propaganda_report.html',  # Report file for propaganda detection
    'metrics_file': 'reports/evaluation_metrics.txt',            # Metrics file for evaluation results
    'misclassification_report_file': 'reports/misclassification_report.txt',  # Misclassification report
    'threshold': 0.5,                                           # Threshold for labeling as 'Propaganda'
    'log_level': 'INFO',  # Default logging level
    'num_topics': 5,       # Number of topics for BERTopic
}

DATABASE = 'users.db'  # Database file for user authentication

# ----------------------------
# Setup Logging
# ----------------------------

def setup_logging(log_file: str, log_level: str):
    """Configure logging to file and console with different levels."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set root logger level to DEBUG

    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # File handler for DEBUG and above
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler with configurable level
    ch = logging.StreamHandler(sys.stdout)
    if log_level.upper() == 'DEBUG':
        ch.setLevel(logging.DEBUG)
    elif log_level.upper() == 'WARNING':
        ch.setLevel(logging.WARNING)
    elif log_level.upper() == 'ERROR':
        ch.setLevel(logging.ERROR)
    elif log_level.upper() == 'CRITICAL':
        ch.setLevel(logging.CRITICAL)
    else:
        ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

# ----------------------------
# Helper Functions
# ----------------------------

def load_propaganda_techniques(file_path: str) -> Dict[str, List[str]]:
    """
    Load propaganda techniques and their associated keywords from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing propaganda techniques and keywords.

    Returns:
        Dict[str, List[str]]: A dictionary with techniques as keys and list of keywords as values.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            techniques = json.load(f)
        if isinstance(techniques, list):
            # If JSON is a list, assume no keywords provided
            techniques_dict = {tech.lower(): [] for tech in techniques}
        elif isinstance(techniques, dict):
            techniques_dict = {tech.lower(): [kw.lower() for kw in kws] for tech, kws in techniques.items()}
        else:
            logging.error(f"Propaganda techniques file '{file_path}' has an unsupported format.")
            st.error(f"Propaganda techniques file '{file_path}' has an unsupported format.")
            sys.exit(1)
        logging.info(f"Loaded {len(techniques_dict)} propaganda techniques from '{file_path}'.")
        return techniques_dict
    except FileNotFoundError:
        logging.error(f"Propaganda techniques file '{file_path}' not found.")
        st.error(f"Propaganda techniques file '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Propaganda techniques file '{file_path}' is not a valid JSON.")
        st.error(f"Propaganda techniques file '{file_path}' is not a valid JSON.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading propaganda techniques: {type(e).__name__} - {e}")
        st.error(f"Error loading propaganda techniques: {type(e).__name__} - {e}")
        sys.exit(1)

def preprocess_text(text: str) -> str:
    """
    Preprocess text by lowercasing and lemmatization using spaCy.

    Args:
        text (str): The raw text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])

def detect_propaganda_techniques_in_text(text: str, techniques: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Detect specific propaganda techniques in the text based on associated keywords.

    Args:
        text (str): The raw text of the article.
        techniques (Dict[str, List[str]]): A dictionary of propaganda techniques with associated keywords.

    Returns:
        Dict[str, List[str]]: A dictionary with techniques as keys and list of detected keywords as values.
    """
    detected_techniques = {}
    text_lower = text.lower()
    for technique, keywords in techniques.items():
        if technique == 'repetitive phrasing':
            continue  # Exclude as per requirements
        detected_keywords = []
        for keyword in keywords:
            # Use regex for exact phrase matching, case-insensitive
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                detected_keywords.append(keyword)
        if detected_keywords:
            detected_techniques[technique] = detected_keywords
    return detected_techniques

def perform_filtered_ner(text: str, relevant_labels: set = RELEVANT_ENTITY_LABELS) -> List[Dict[str, str]]:
    """
    Perform Named Entity Recognition and filter entities by relevance.

    Args:
        text (str): The raw text of the article.
        relevant_labels (set): A set of entity labels to retain.

    Returns:
        List[Dict[str, str]]: A list of relevant detected entities with their labels.
    """
    try:
        doc = ner_model(text)
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents if ent.label_ in relevant_labels]
        return entities
    except Exception as e:
        logging.error(f"NER processing error for text: {text[:50]}... - {type(e).__name__}: {e}")
        return []

def load_transformer_model(model_path: str, label_encoder_path: str):
    """Load the trained transformer model and label encoder."""
    try:
        logging.info("Loading transformer model and tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set model to evaluation mode
        logging.info(f"Loaded transformer model from '{model_path}'.")

        logging.info("Loading label encoder...")
        mlb = joblib.load(label_encoder_path)
        logging.info(f"Loaded label encoder from '{label_encoder_path}'.")

        return model, tokenizer, mlb
    except FileNotFoundError as e:
        logging.error(f"Model or label encoder file not found: {e}")
        st.error(f"Model or label encoder file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        sys.exit(1)

def detect_media_bias(text: str, model, tokenizer, mlb, threshold: float = 0.5) -> str:
    """
    Run media bias detection on a given text.

    Args:
        text (str): The input text to detect bias.
        model: The pre-trained model for media bias detection.
        tokenizer: The tokenizer for the bias detection model.
        mlb: MultiLabelBinarizer for bias categories.
        threshold (float): Threshold for classification.

    Returns:
        str: Predicted bias category or score.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).numpy()[0]
            label_idx = (probs > threshold).astype(int)
            predicted = mlb.inverse_transform([label_idx])[0]
            bias_category = predicted[0] if predicted else "Neutral"
        return bias_category
    except Exception as e:
        logging.error(f"Error during media bias detection for text: {text[:50]}... - {type(e).__name__} - {e}")
        return "Unknown"

def predict_with_transformer(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    mlb: MultiLabelBinarizer,
    texts: List[str],
    threshold: float = 0.5
) -> List[List[str]]:
    """
    Predict propaganda techniques using the trained transformer model.

    Args:
        model (DistilBertForSequenceClassification): The trained transformer model.
        tokenizer (DistilBertTokenizer): The tokenizer.
        mlb (MultiLabelBinarizer): The label binarizer.
        texts (List[str]): A list of preprocessed texts.
        threshold (float): Threshold for classification.

    Returns:
        List[List[str]]: A list of predicted propaganda techniques for each text.
    """
    try:
        logging.info("Starting prediction with transformer model...")
        predictions = []
        model.to('cpu')  # Ensure model is on CPU for prediction
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).numpy()[0]
                labels = (probs > threshold).astype(int)
                predicted = mlb.inverse_transform([labels])[0]
                predictions.append(list(predicted))
        logging.info("Completed predictions with transformer model.")
        return predictions
    except Exception as e:
        logging.error(f"Error in transformer prediction: {type(e).__name__} - {e}")
        st.error(f"Error in transformer prediction: {type(e).__name__} - {e}")
        return [[] for _ in texts]

def generate_evaluation_metrics(df: pd.DataFrame, mlb: MultiLabelBinarizer, metrics_file: str):
    """
    Automatically calculate and save classification metrics.

    Args:
        df (pd.DataFrame): The DataFrame containing true and predicted labels.
        mlb (MultiLabelBinarizer): The label binarizer.
        metrics_file (str): Path to save the metrics report.
    """
    try:
        logging.info("Calculating automated evaluation metrics...")
        y_true = mlb.transform(df['Techniques'])
        y_pred = mlb.transform(df['Predicted_Techniques'])

        accuracy = accuracy_score(y_true, y_pred)
        hamming = hamming_loss(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro')

        # Save metrics to a text file
        metrics_dir = os.path.dirname(metrics_file)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        with open(metrics_file, 'w') as f:
            f.write("Automated Evaluation Metrics\n")
            f.write("============================\n\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Hamming Loss: {hamming:.4f}\n")
            f.write(f"Average F1 Score: {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)

        logging.info(f"Automated evaluation metrics saved to '{metrics_file}'.")
    except Exception as e:
        logging.error(f"Error calculating evaluation metrics: {type(e).__name__} - {e}")
        st.error(f"Error calculating evaluation metrics: {type(e).__name__} - {e}")

def perform_detailed_misclassification_analysis(
    y_true,
    y_pred,
    mlb: MultiLabelBinarizer,
    report_file: str
):
    """
    Analyze misclassifications to identify common errors, including top misclassified techniques.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        mlb (MultiLabelBinarizer): Label binarizer.
        report_file (str): Path to save the misclassification report.
    """
    try:
        logging.info("Starting detailed misclassification analysis...")
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)

        for true, pred in zip(y_true, y_pred):
            for idx, (t, p) in enumerate(zip(true, pred)):
                if p and not t:
                    false_positives[mlb.classes_[idx]] += 1
                elif t and not p:
                    false_negatives[mlb.classes_[idx]] += 1

        # Identify top misclassified techniques
        top_false_positives = sorted(false_positives.items(), key=lambda x: x[1], reverse=True)[:5]
        top_false_negatives = sorted(false_negatives.items(), key=lambda x: x[1], reverse=True)[:5]

        # Save the analysis to a file
        misclassification_dir = os.path.dirname(report_file)
        if misclassification_dir and not os.path.exists(misclassification_dir):
            os.makedirs(misclassification_dir)

        with open(report_file, 'w') as f:
            f.write("Detailed Misclassification Analysis Report\n")
            f.write("==========================================\n\n")
            f.write("Top False Positives:\n")
            for technique, count in top_false_positives:
                f.write(f"- {technique}: {count}\n")
            f.write("\nTop False Negatives:\n")
            for technique, count in top_false_negatives:
                f.write(f"- {technique}: {count}\n")

        logging.info(f"Detailed misclassification analysis saved to '{report_file}'.")
    except Exception as e:
        logging.error(f"Error during detailed misclassification analysis: {type(e).__name__} - {e}")
        st.error(f"Error during detailed misclassification analysis: {type(e).__name__} - {e}")

# ----------------------------
# Authentication Functions
# ----------------------------

def create_connection():
    try:
        conn = sqlite3.connect(DATABASE)
        logging.info("Connected to the database.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection failed: {e}")
        sys.exit(1)

def ensure_users_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        logging.info("Ensured users table exists.")
    except sqlite3.Error as e:
        logging.error(f"Failed to create users table: {e}")
        sys.exit(1)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email, password):
    conn = create_connection()
    ensure_users_table(conn)
    hashed_password = hash_password(password)
    try:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed_password))
        conn.commit()
        logging.info(f"User '{email}' registered successfully.")
        return True
    except sqlite3.IntegrityError:
        logging.warning(f"User '{email}' already exists.")
        return False
    except sqlite3.Error as e:
        logging.error(f"Database error during registration: {e}")
        return False
    finally:
        conn.close()

def verify_user(email, password):
    conn = create_connection()
    ensure_users_table(conn)
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT password FROM users WHERE email = ?', (email,))
        result = cursor.fetchone()
        if result:
            stored_password = result[0]
            if stored_password == hash_password(password):
                logging.info(f"User '{email}' authenticated successfully.")
                return True
            else:
                logging.warning(f"Incorrect password for user '{email}'.")
                return False
        else:
            logging.info(f"User '{email}' not found.")
            return False
    except sqlite3.Error as e:
        logging.error(f"Database error during login: {e}")
        return False
    finally:
        conn.close()

# ----------------------------
# Streamlit UI Components
# ----------------------------

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Propaganda and Media Bias Detection",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize Logging
    setup_logging(DEFAULT_CONFIG['log_file'], DEFAULT_CONFIG['log_level'])

    # Ensure necessary directories exist
    for directory in ['models', 'reports', 'logs']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

    # Initialize Session State for Authentication
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = ''

    # Authentication
    menu = ["Home", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("📊 Propaganda and Media Bias Detection in Articles")
        if st.session_state['authenticated']:
            st.success(f"Logged in as {st.session_state['user_email']}")
            # Display the main app content here
            app_content()
        else:
            st.info("Please log in to use the application.")

    elif choice == "Login":
        st.subheader("Login to Your Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')

        if st.button("Login"):
            if verify_user(email, password):
                st.session_state['authenticated'] = True
                st.session_state['user_email'] = email
                st.success(f"Logged in as {email}")
                st.experimental_rerun()
            else:
                st.error("Invalid email or password.")

    elif choice == "SignUp":
        st.subheader("Create a New Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        confirm_password = st.text_input("Confirm Password", type='password')

        if st.button("SignUp"):
            if password == confirm_password:
                if register_user(email, password):
                    st.success("Account created successfully. Please log in.")
                else:
                    st.warning("An account with this email already exists.")
            else:
                st.error("Passwords do not match.")

def app_content():
    st.title("📊 Propaganda and Media Bias Detection in Articles")

    # Sidebar for Configuration
    st.sidebar.title("🔧 Configuration")

    # File Uploader for JSON Data
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a JSON file containing articles", type="json")

    # Slider for Threshold
    threshold = st.sidebar.slider("Threshold for Propaganda Detection", 0.0, 1.0, DEFAULT_CONFIG['threshold'], 0.05)

    # Predict Button
    analyze_button = st.sidebar.button("Analyze Articles")

    # Text Area for Custom Text Analysis
    st.header("📝 Analyze Custom Text")
    user_input = st.text_area("Enter text to analyze for propaganda and bias:")

    analyze_text_button = st.button("Analyze Text")

    # Handle Uploaded Data
    if analyze_button:
        if uploaded_file is not None:
            try:
                # Load data
                data = json.load(uploaded_file)
                df = pd.json_normalize(data)
                st.success(f"Loaded data with {len(df)} records.")

                # Ensure 'content' column exists
                if 'content' not in df.columns and 'Title' in df.columns:
                    df.rename(columns={'Title': 'content'}, inplace=True)
                    st.info("Renamed 'Title' column to 'content'.")
                elif 'content' not in df.columns:
                    st.error("The 'content' column is missing from the dataset.")
                    st.stop()

                # Extract techniques from 'Indicators'
                def extract_techniques(indicators):
                    if isinstance(indicators, list):
                        return [indicator.get('Propaganda Technique') for indicator in indicators]
                    else:
                        return []

                df['Techniques'] = df['Indicators'].apply(extract_techniques)

                # Remove entries without techniques
                df = df[df['Techniques'].apply(len) > 0]

                # Preprocess text
                with st.spinner("Preprocessing text data..."):
                    df['processed_content'] = df['content'].astype(str).swifter.apply(preprocess_text)
                st.success("Completed text preprocessing.")

                # Perform NER
                with st.spinner("Performing Named Entity Recognition (NER)..."):
                    df['Entities'] = df['content'].astype(str).swifter.apply(perform_filtered_ner)
                st.success("Completed Named Entity Recognition.")

                # Prepare labels
                df['Is_Propaganda'] = df['Techniques'].apply(lambda x: True if x else False)

                # Load trained model and label encoder
                with st.spinner("Loading trained transformer model and label encoder..."):
                    propaganda_model, propaganda_tokenizer, propaganda_mlb = load_transformer_model(
                        DEFAULT_CONFIG['propaganda_model_file'],
                        DEFAULT_CONFIG['label_encoder_file']
                    )
                st.success("Loaded transformer model and label encoder.")

                # Predict Propaganda Techniques
                with st.spinner("Detecting propaganda techniques in articles..."):
                    df['Predicted_Techniques'] = predict_with_transformer(
                        propaganda_model,
                        propaganda_tokenizer,
                        propaganda_mlb,
                        df['processed_content'].tolist(),
                        threshold=threshold
                    )
                st.success("Completed propaganda techniques detection.")

                # Generate Evaluation Metrics
                with st.spinner("Calculating evaluation metrics..."):
                    generate_evaluation_metrics(df, propaganda_mlb, DEFAULT_CONFIG['metrics_file'])
                st.success("Evaluation metrics calculated.")

                # Perform Misclassification Analysis
                with st.spinner("Performing misclassification analysis..."):
                    y_true = propaganda_mlb.transform(df['Techniques'])
                    y_pred = propaganda_mlb.transform(df['Predicted_Techniques'])
                    perform_detailed_misclassification_analysis(
                        y_true=y_true,
                        y_pred=y_pred,
                        mlb=propaganda_mlb,
                        report_file=DEFAULT_CONFIG['misclassification_report_file']
                    )
                st.success("Misclassification analysis completed.")

                # Media Bias Detection
                with st.spinner("Loading media bias detection model..."):
                    bias_model, bias_tokenizer, bias_mlb = load_transformer_model(
                        DEFAULT_CONFIG['bias_model_file'],
                        DEFAULT_CONFIG['label_encoder_file']
                    )
                st.success("Loaded media bias detection model.")

                # Predict Media Bias
                with st.spinner("Detecting media bias in articles..."):
                    df['Bias_Category'] = df['processed_content'].apply(
                        lambda x: detect_media_bias(x, bias_model, bias_tokenizer, bias_mlb, threshold=0.5)
                    )
                st.success("Completed media bias detection.")

                # Display Results
                st.subheader("📊 Annotated Data")
                st.dataframe(df.head())

                st.subheader("🗂 Propaganda Techniques Detected")
                st.write(df[['content', 'Predicted_Techniques']].head())

                st.subheader("🧠 Media Bias Detection")
                st.write(df[['content', 'Bias_Category']].head())

                # Provide Download Button
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Annotated Data as CSV",
                    data=csv_data,
                    file_name='annotated_data.csv',
                    mime='text/csv',
                )

                # Display Evaluation Metrics
                st.subheader("📋 Evaluation Metrics")
                with open(DEFAULT_CONFIG['metrics_file'], 'r') as f:
                    metrics_content = f.read()
                st.text(metrics_content)

                # Display Misclassification Report
                st.subheader("🔍 Misclassification Analysis")
                with open(DEFAULT_CONFIG['misclassification_report_file'], 'r') as f:
                    misclass_content = f.read()
                st.text(misclass_content)

            except Exception as e:
                logging.error(f"An error occurred during analysis: {type(e).__name__} - {e}")
                st.error(f"An error occurred during analysis: {type(e).__name__} - {e}")
        else:
            st.warning("Please upload the articles JSON file.")

    # Handle Custom Text Analysis
    if analyze_text_button:
        if user_input:
            try:
                with st.spinner("Preprocessing your text..."):
                    processed_text = preprocess_text(user_input)
                st.success("Completed text preprocessing.")

                # Load and predict using propaganda model
                with st.spinner("Loading transformer model and label encoder..."):
                    propaganda_model, propaganda_tokenizer, propaganda_mlb = load_transformer_model(
                        DEFAULT_CONFIG['propaganda_model_file'],
                        DEFAULT_CONFIG['label_encoder_file']
                    )
                st.success("Loaded transformer model and label encoder.")

                with st.spinner("Predicting propaganda techniques..."):
                    predicted_techniques = predict_with_transformer(
                        propaganda_model,
                        propaganda_tokenizer,
                        propaganda_mlb,
                        [processed_text],
                        threshold=threshold
                    )[0]
                if predicted_techniques:
                    st.success("Predicted Propaganda Techniques:")
                    st.markdown(", ".join(predicted_techniques))
                else:
                    st.info("No propaganda techniques predicted.")

                # Load and predict media bias
                with st.spinner("Loading media bias detection model..."):
                    bias_model, bias_tokenizer, bias_mlb = load_transformer_model(
                        DEFAULT_CONFIG['bias_model_file'],
                        DEFAULT_CONFIG['label_encoder_file']
                    )
                st.success("Loaded media bias detection model.")

                with st.spinner("Predicting media bias..."):
                    bias_category = detect_media_bias(user_input, bias_model, bias_tokenizer, bias_mlb, threshold=0.5)
                st.success(f"Predicted Media Bias Category: **{bias_category}**")

                # Perform NER
                with st.spinner("Performing Named Entity Recognition (NER)..."):
                    entities = perform_filtered_ner(user_input)
                if entities:
                    st.success("Named Entities detected:")
                    for ent in entities:
                        st.markdown(f"{ent['text']} ({ent['label']})")
                else:
                    st.info("No relevant entities detected.")

            except Exception as e:
                logging.error(f"An error occurred during text analysis: {type(e).__name__} - {e}")
                st.error(f"An error occurred during text analysis: {type(e).__name__} - {e}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
