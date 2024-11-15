# app.py

import os
import sys
import json
import logging
from typing import List, Tuple, Dict
import pandas as pd
import swifter
import re
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    hamming_loss,
    f1_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from bertopic import BERTopic
import joblib
import torch
import streamlit as st
from tqdm import tqdm
from collections import defaultdict

# Import the authentication module
import auth

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
    'propaganda_techniques_file': 'models/propaganda_techniques.json',  # Techniques JSON file
    'bias_model_file': 'models/media_bias_model',                        # Media bias model path
    'propaganda_model_file': 'models/propaganda_detection_model',         # Propaganda model path
    'label_encoder_file': 'models/label_encoder.pkl',                     # Label encoder path
    'media_bias_label_encoder_file': 'models/bias_label_encoder.pkl',     # Media bias label encoder path
    'log_file': 'logs/propaganda_detection.log',                          # Log file path
    'bias_report_file': 'reports/media_bias_report.html',                 # Report file for media bias detection
    'propaganda_report_file': 'reports/propaganda_report.html',           # Report file for propaganda detection
    'metrics_file': 'reports/evaluation_metrics.txt',                     # Metrics file for evaluation results
    'misclassification_report_file': 'reports/misclassification_report.txt', # Misclassification report
    'threshold': 0.5,                                                    # Threshold for labeling as Propaganda
    'num_topics': 5,                                                      # Number of topics for BERTopic
    'log_level': 'INFO',                                                  # Default logging level
}

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

# Initialize Logging
setup_logging(DEFAULT_CONFIG['log_file'], DEFAULT_CONFIG['log_level'])

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

def perform_topic_modeling(texts: List[str], num_topics: int) -> Tuple[pd.DataFrame, pd.DataFrame, BERTopic]:
    """
    Perform BERTopic on the corpus.

    Args:
        texts (List[str]): A list of preprocessed texts.
        num_topics (int): The number of topics to generate.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, BERTopic]: 
            - topic_info: DataFrame containing topic information.
            - topic_freq: DataFrame containing topic frequencies.
            - topic_model: The BERTopic model instance.
    """
    try:
        logging.info("Starting BERTopic modeling...")
        topic_model = BERTopic(n_gram_range=(1, 2), min_topic_size=10, nr_topics=num_topics)
        topics, probs = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()
        logging.info(f"Generated {num_topics} topics using BERTopic.")
        logging.info("BERTopic modeling completed successfully.")

        # Verify that the number of topics matches the number of documents
        topics = list(topics)  # Ensure topics is a list
        if len(topics) != len(texts):
            logging.error(f"Mismatch in number of topics ({len(topics)}) and number of documents ({len(texts)}). Adjusting list length.")
            st.error(f"Mismatch in number of topics ({len(topics)}) and number of documents ({len(texts)}). Adjusting list length.")
            if len(topics) > len(texts):
                topics = topics[:len(texts)]
                logging.warning(f"Trimmed topics list to match the number of documents ({len(texts)}).")
            else:
                topics.extend([None] * (len(texts) - len(topics)))  # Use extend to add None values
                logging.warning(f"Padded topics list with 'None' to match the number of documents ({len(texts)}).")

        topic_freq = topic_model.get_topic_freq()
        return topic_info, topic_freq, topic_model
    except Exception as e:
        logging.error(f"Error in topic modeling: {type(e).__name__} - {e}")
        st.error(f"Error in topic modeling: {type(e).__name__} - {e}")
        # Return empty DataFrames and None for topic_model_instance
        return pd.DataFrame(), pd.DataFrame(), None

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

def generate_enhanced_interactive_report(df: pd.DataFrame, topic_info: pd.DataFrame, report_file: str):
    """
    Generate an enhanced interactive HTML report with detailed propaganda techniques information.

    Args:
        df (pd.DataFrame): The DataFrame containing annotated article data.
        topic_info (pd.DataFrame): The DataFrame containing topic information.
        report_file (str): Path where the HTML report will be saved.
    """
    try:
        logging.info("Generating enhanced interactive report...")

        # Summary Statistics
        propaganda_count = df['Is_Propaganda'].value_counts().to_dict()

        # Plot Propaganda Distribution
        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x='Is_Propaganda')
        plt.title('Propaganda Distribution')
        plot_path = 'reports/propaganda_distribution.png'
        plt.savefig(plot_path)
        plt.close()
        logging.info("Saved propaganda distribution plot.")

        # Topics
        if not topic_info.empty:
            topics_html = "<br>".join([f"Topic {row['Topic']}: {row['Name']}" for index, row in topic_info.iterrows()])
        else:
            topics_html = "No topics were identified due to an error in topic modeling."

        # Techniques Frequency
        techniques_freq = df['Predicted_Techniques'].explode().value_counts().to_dict()
        techniques_freq_html = "<ul>" + "".join([f"<li>{technique}: {count}</li>" for technique, count in techniques_freq.items()]) + "</ul>"

        # Generate HTML Report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"""
            <html>
            <head><title>Enhanced Propaganda Detection Report</title></head>
            <body>
                <h1>Enhanced Propaganda Detection Report</h1>
                <h2>Summary</h2>
                <p><strong>Propaganda Articles:</strong> {propaganda_count.get(True, 0)}</p>
                <p><strong>Non-Propaganda Articles:</strong> {propaganda_count.get(False, 0)}</p>
                <img src="{plot_path}" alt="Propaganda Distribution">
                <h2>Topic Modeling</h2>
                <p>{topics_html}</p>
                <h2>Propaganda Techniques Detected</h2>
                {techniques_freq_html}
                <h2>Detailed Techniques in Articles</h2>
                {df[['article_id', 'Detected_Techniques']].to_html(index=False)}
            </body>
            </html>
            """)
        logging.info(f"Enhanced interactive report generated at '{report_file}'.")
    except Exception as e:
        logging.error(f"Error generating enhanced interactive report: {type(e).__name__} - {e}")
        st.error(f"Error generating enhanced interactive report: {type(e).__name__} - {e}")

def load_transformer_model(model_path: str, label_encoder_path: str):
    """Load the trained transformer model and label encoder."""
    return auth.load_transformer_model(model_path, label_encoder_path)

def load_bias_model(model_path: str):
    """Load the media bias detection model."""
    return auth.load_bias_model(model_path)

def load_media_bias_label_encoder(label_encoder_path: str):
    """Load or initialize the media bias label encoder."""
    return auth.load_media_bias_label_encoder(label_encoder_path)

def detect_media_bias(text: str, model, tokenizer, mlb, threshold: float = 0.5) -> str:
    """Detect media bias in the given text."""
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

def generate_media_bias_report(df: pd.DataFrame, report_file: str):
    """Generate an HTML report summarizing media bias results."""
    try:
        logging.info("Generating media bias report...")

        # Summarize bias distribution
        bias_summary = df['Bias_Category'].value_counts().to_dict()
        summary_html = "".join([f"<li>{k}: {v}</li>" for k, v in bias_summary.items()])

        # Generate HTML Report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"""
            <html>
            <head><title>Media Bias Detection Report</title></head>
            <body>
                <h1>Media Bias Detection Report</h1>
                <h2>Bias Distribution</h2>
                <ul>{summary_html}</ul>
                <h2>Detailed Bias in Articles</h2>
                {df[['article_id', 'Bias_Category']].to_html(index=False)}
            </body>
            </html>
            """)
        logging.info(f"Media bias report generated at '{report_file}'.")
    except Exception as e:
        logging.error(f"Error generating media bias report: {type(e).__name__} - {e}")
        st.error(f"Error generating media bias report: {type(e).__name__} - {e}")

def generate_evaluation_metrics(df: pd.DataFrame, mlb: MultiLabelBinarizer, metrics_file: str):
    """Calculate and save classification metrics."""
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
    """Analyze misclassifications to identify common errors."""
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
# Machine Learning Functions
# ----------------------------

def train_transformer_model(df: pd.DataFrame, model_save_path: str, label_encoder_file: str):
    """
    Train a transformer-based multi-label classification model for propaganda techniques detection.

    Args:
        df (pd.DataFrame): The DataFrame containing annotated article data.
        model_save_path (str): Path where the trained model will be saved.
        label_encoder_file (str): Path where the label encoder will be saved.
    """
    try:
        logging.info("Starting transformer model training...")

        # Initialize tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(df['Techniques'].explode().unique()),
            problem_type="multi_label_classification"
        )

        # Prepare data
        dataset = df[['processed_content', 'Techniques']].rename(columns={'processed_content': 'text', 'Techniques': 'labels'})
        texts = dataset['text'].tolist()
        labels = dataset['labels'].tolist()

        # Tokenize the data
        encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

        # Binarize labels
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(labels)
        joblib.dump(mlb, label_encoder_file)
        logging.info(f"Label encoder saved to '{label_encoder_file}'.")

        # Create Dataset
        class PropagandaDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
                return item

            def __len__(self):
                return len(self.labels)

        prop_dataset = PropagandaDataset(encodings, y)

        # Split into train and test
        train_size = int(0.8 * len(prop_dataset))
        test_size = len(prop_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(prop_dataset, [train_size, test_size])

        # Define training parameters
        epochs = 3
        batch_size = 8

        # Training Loop
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            model.train()
            total_loss = 0
            for i in tqdm(range(0, len(train_dataset), batch_size), desc="Training"):
                batch = train_dataset[i:i+batch_size]
                input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
                attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
                labels = torch.stack([item['labels'] for item in batch]).to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / (len(train_dataset) / batch_size)
            logging.info(f"Average training loss: {avg_loss:.4f}")

        # Save the model
        model_dir = os.path.dirname(model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logging.info(f"Transformer model saved to '{model_save_path}'.")

        # Evaluate the model
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
                batch = test_dataset[i:i+batch_size]
                input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
                attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
                labels = torch.stack([item['labels'] for item in batch]).cpu().numpy()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions = (probs > 0.5).astype(int)

                y_true.extend(labels)
                y_pred.extend(predictions)

        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=mlb.classes_, zero_division=0)
        logging.info(f"Classification Report:\n{report}")

        # Calculate average F1 score
        f1 = f1_score(y_true, y_pred, average='macro')
        logging.info(f"Average F1 Score: {f1:.4f}")

        # Save Evaluation Metrics
        metrics_dir = os.path.dirname(DEFAULT_CONFIG['metrics_file'])
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        with open(DEFAULT_CONFIG['metrics_file'], 'w') as f:
            f.write("Classification Report:\n")
            f.write(report)
            f.write(f"\nAverage F1 Score: {f1:.4f}\n")
        logging.info(f"Evaluation metrics saved to '{DEFAULT_CONFIG['metrics_file']}'.")

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
            for text in tqdm(texts, desc="Predicting"):
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
            return [[] for _ in texts]

def perform_semi_supervised_learning(
    df: pd.DataFrame,
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizer,
    mlb: MultiLabelBinarizer,
    threshold: float = 0.9
) -> pd.DataFrame:
    """
    Perform semi-supervised learning by labeling unlabeled data with high-confidence predictions.

    Args:
        df (pd.DataFrame): The DataFrame containing article data.
        model (DistilBertForSequenceClassification): The trained transformer model.
        tokenizer (DistilBertTokenizer): The tokenizer.
        mlb (MultiLabelBinarizer): The label binarizer.
        threshold (float): Confidence threshold for selecting pseudo-labeled data.

    Returns:
        pd.DataFrame: The augmented DataFrame with pseudo-labeled data.
    """
    try:
        logging.info("Starting semi-supervised learning process...")

        # Identify unlabeled data (assuming 'Techniques' is empty or NaN)
        unlabeled_df = df[df['Techniques'].isnull() | (df['Techniques'].apply(len) == 0)]
        logging.info(f"Found {len(unlabeled_df)} unlabeled articles.")

        if unlabeled_df.empty:
            logging.info("No unlabeled data found. Skipping semi-supervised learning.")
            return df

        # Predict techniques on unlabeled data
        predictions = predict_with_transformer(model, tokenizer, mlb, unlabeled_df['processed_content'].tolist(), threshold=threshold)

        # Select high-confidence predictions (assuming threshold used in prediction is sufficient)
        pseudo_labeled_df = unlabeled_df.copy()
        pseudo_labeled_df['Techniques'] = predictions
        pseudo_labeled_df = pseudo_labeled_df[pseudo_labeled_df['Techniques'].apply(len) > 0]
        logging.info(f"Selected {len(pseudo_labeled_df)} high-confidence pseudo-labeled articles.")

        if pseudo_labeled_df.empty:
            logging.info("No high-confidence pseudo-labeled data found. Skipping augmentation.")
            return df

        # Append pseudo-labeled data to the original DataFrame
        augmented_df = pd.concat([df, pseudo_labeled_df], ignore_index=True)
        logging.info("Augmented training data with pseudo-labeled articles.")

        return augmented_df
    except Exception as e:
        logging.error(f"Error during semi-supervised learning: {type(e).__name__} - {e}")
        return df

# ----------------------------
# Streamlit UI Components
# ----------------------------

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="📊 Propaganda and Media Bias Detection",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # App title
    st.title("📊 Propaganda and Media Bias Detection App")

    # Sidebar navigation menu
    menu = ["Home", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home page
    if choice == "Home":
        st.subheader("Welcome to the Propaganda Detection App")
        st.markdown("""
            This application allows you to analyze articles for propaganda techniques and media bias.
            Please register or log in to get started.
        """)

    # Login page
    elif choice == "Login":
        if st.session_state['logged_in']:
            st.info(f"Logged in as {st.session_state['user_email']}")
            if st.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['user_email'] = ''
                st.success("Logged out successfully.")
        else:
            st.subheader("Login to Your Account")
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')

            if st.button("Login"):
                if auth.verify_user(email, password):
                    st.session_state['logged_in'] = True
                    st.session_state['user_email'] = email
                    st.success(f"Logged in as {email}")
                else:
                    st.error("Invalid email or password.")

    # SignUp page
    elif choice == "SignUp":
        if st.session_state['logged_in']:
            st.info(f"Already logged in as {st.session_state['user_email']}")
            if st.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['user_email'] = ''
                st.success("Logged out successfully.")
        else:
            st.subheader("Create a New Account")
            email = st.text_input("Email", key='signup_email')
            password = st.text_input("Password", type='password', key='signup_password')
            confirm_password = st.text_input("Confirm Password", type='password', key='signup_confirm_password')

            if st.button("SignUp"):
                if not email or not password or not confirm_password:
                    st.error("Please fill out all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    if auth.register_user(email, password):
                        st.success("Account created successfully. You can now log in.")
                    else:
                        st.warning("An account with this email already exists.")

    # After Login: Main Functionalities
    if st.session_state['logged_in']:
        st.header("🔍 Analyze Articles for Propaganda and Media Bias")

        # Sidebar for Configuration
        st.sidebar.title("🔧 Configuration")

        # File Uploader for Articles JSON
        uploaded_file = st.sidebar.file_uploader("📂 Upload Articles JSON File", type="json")

        # File Uploader for Propaganda Techniques JSON
        techniques_file = st.sidebar.file_uploader("📂 Upload Propaganda Techniques JSON File", type="json", key='techniques')

        # Slider for Threshold
        threshold = st.sidebar.slider("⚖️ Threshold for Propaganda Detection", 0.0, 1.0, DEFAULT_CONFIG['threshold'], 0.05)

        # Number input for Number of Topics
        num_topics = st.sidebar.number_input("🧠 Number of Topics for BERTopic", min_value=1, max_value=20, value=DEFAULT_CONFIG['num_topics'])

        # Button to Start Analysis
        analyze_button = st.sidebar.button("🔍 Analyze Articles")

        # Text Area for Custom Text Analysis
        st.subheader("📝 Analyze Custom Text")
        user_input = st.text_area("Enter text to analyze for propaganda and bias:")

        analyze_text_button = st.button("🔍 Analyze Text")

        # Handle Uploaded Data Analysis
        if analyze_button:
            if uploaded_file and techniques_file:
                try:
                    # Load Articles Data
                    data = json.load(uploaded_file)
                    df = pd.json_normalize(data)
                    st.success(f"✅ Loaded data with {len(df)} records.")

                    # Ensure 'content' column exists
                    if 'content' not in df.columns and 'Title' in df.columns:
                        df.rename(columns={'Title': 'content'}, inplace=True)
                        st.info("🔄 Renamed 'Title' column to 'content'.")
                    elif 'content' not in df.columns:
                        st.error("❌ The 'content' column is missing from the dataset.")
                        st.stop()

                    # Save Propaganda Techniques JSON to file
                    techniques_path = DEFAULT_CONFIG['propaganda_techniques_file']
                    with open(techniques_path, 'w', encoding='utf-8') as f:
                        json.dump(json.load(techniques_file), f, ensure_ascii=False, indent=4)
                    st.success("✅ Propaganda techniques file uploaded successfully.")

                    # Load Propaganda Techniques
                    techniques = load_propaganda_techniques(techniques_path)

                    # Preprocess Text
                    with st.spinner("🧹 Preprocessing text data..."):
                        df['processed_content'] = df['content'].astype(str).swifter.apply(preprocess_text)
                    st.success("✅ Completed text preprocessing.")

                    # Detect Propaganda Techniques
                    with st.spinner("🔍 Detecting propaganda techniques in articles..."):
                        df['Detected_Techniques'] = df['content'].astype(str).swifter.apply(
                            lambda x: detect_propaganda_techniques_in_text(x, techniques)
                        )
                    st.success("✅ Completed propaganda techniques detection.")

                    # Perform NER
                    with st.spinner("🧠 Performing Named Entity Recognition (NER)..."):
                        df['Entities'] = df['content'].astype(str).swifter.apply(perform_filtered_ner)
                    st.success("✅ Completed Named Entity Recognition.")

                    # Prepare Labels
                    df['Is_Propaganda'] = df['Detected_Techniques'].apply(lambda x: True if x else False)
                    df['Techniques'] = df['Detected_Techniques']

                    # Train Transformer Model
                    with st.spinner("🏋️‍♂️ Training transformer model... This may take a while."):
                        train_transformer_model(df, DEFAULT_CONFIG['propaganda_model_file'], DEFAULT_CONFIG['label_encoder_file'])
                    st.success("✅ Transformer model trained successfully.")

                    # Load Trained Model and Label Encoder
                    with st.spinner("📦 Loading trained transformer model and label encoder..."):
                        propaganda_model, propaganda_tokenizer, propaganda_mlb = load_transformer_model(
                            DEFAULT_CONFIG['propaganda_model_file'],
                            DEFAULT_CONFIG['label_encoder_file']
                        )
                    st.success("✅ Loaded transformer model and label encoder.")

                    # Predict Propaganda Techniques
                    with st.spinner("🔍 Detecting propaganda techniques in articles..."):
                        df['Predicted_Techniques'] = predict_with_transformer(
                            propaganda_model,
                            propaganda_tokenizer,
                            propaganda_mlb,
                            df['processed_content'].tolist(),
                            threshold=0.5
                        )
                    st.success("✅ Completed propaganda techniques detection.")

                    # Topic Modeling
                    with st.spinner("🧠 Performing topic modeling with BERTopic..."):
                        topic_info, topic_freq, topic_model_instance = perform_topic_modeling(df['processed_content'].tolist(), num_topics)
                    st.success("✅ Completed topic modeling.")

                    # Media Bias Detection
                    with st.spinner("🔍 Loading media bias detection model..."):
                        bias_model, bias_tokenizer = load_bias_model(DEFAULT_CONFIG['bias_model_file'])
                    st.success("✅ Loaded media bias detection model.")

                    # Load or Define MultiLabelBinarizer for Media Bias Categories
                    bias_mlb_file = DEFAULT_CONFIG['media_bias_label_encoder_file']
                    bias_mlb = load_media_bias_label_encoder(bias_mlb_file)

                    # Predict Media Bias
                    with st.spinner("🔍 Detecting media bias in articles..."):
                        df['Bias_Category'] = df['processed_content'].apply(
                            lambda x: detect_media_bias(x, bias_model, bias_tokenizer, bias_mlb, threshold=0.5)
                        )
                    st.success("✅ Completed media bias detection.")

                    # Generate Reports
                    with st.spinner("📄 Generating reports..."):
                        generate_enhanced_interactive_report(df, topic_info, DEFAULT_CONFIG['propaganda_report_file'])
                        generate_media_bias_report(df, DEFAULT_CONFIG['bias_report_file'])
                    st.success("✅ Reports generated successfully.")

                    # Generate Evaluation Metrics
                    with st.spinner("📈 Calculating evaluation metrics..."):
                        generate_evaluation_metrics(df, propaganda_mlb, DEFAULT_CONFIG['metrics_file'])
                    st.success("✅ Evaluation metrics calculated.")

                    # Perform Misclassification Analysis
                    with st.spinner("🔍 Performing misclassification analysis..."):
                        perform_detailed_misclassification_analysis(
                            y_true=propaganda_mlb.transform(df['Techniques']),
                            y_pred=propaganda_mlb.transform(df['Predicted_Techniques']),
                            mlb=propaganda_mlb,
                            report_file=DEFAULT_CONFIG['misclassification_report_file']
                        )
                    st.success("✅ Misclassification analysis completed.")

                    # Display Results
                    st.subheader("📊 Annotated Data")
                    st.dataframe(df.head())

                    st.subheader("📈 Propaganda Distribution")
                    fig, ax = plt.subplots()
                    sns.countplot(data=df, x='Is_Propaganda', ax=ax)
                    plt.title('Propaganda Distribution')
                    st.pyplot(fig)

                    st.subheader("🧠 Topic Modeling")
                    if not topic_info.empty:
                        st.write(topic_info)
                    else:
                        st.write("No topics were identified.")

                    st.subheader("🗂 Propaganda Techniques Detected")
                    st.write(df[['article_id', 'Predicted_Techniques']].head())

                    st.subheader("🧠 Media Bias Detection")
                    st.write(df[['article_id', 'Bias_Category']].head())

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
                st.warning("⚠️ Please upload both the articles JSON file and the propaganda techniques JSON file.")

        # Handle Custom Text Analysis
        if analyze_text_button:
            if user_input:
                try:
                    with st.spinner("🧹 Preprocessing your text..."):
                        processed_text = preprocess_text(user_input)
                    st.success("✅ Completed text preprocessing.")

                    # Load Propaganda Techniques
                    techniques_path = DEFAULT_CONFIG['propaganda_techniques_file']
                    techniques = load_propaganda_techniques(techniques_path)

                    # Detect Propaganda Techniques
                    with st.spinner("🔍 Detecting propaganda techniques..."):
                        detected_techniques = detect_propaganda_techniques_in_text(user_input, techniques)
                    if detected_techniques:
                        st.success("✅ Propaganda techniques detected:")
                        for technique, keywords in detected_techniques.items():
                            st.markdown(f"**{technique.title()}**: {', '.join(keywords)}")
                    else:
                        st.info("ℹ️ No propaganda techniques detected.")

                    # Perform NER
                    with st.spinner("🧠 Performing Named Entity Recognition (NER)..."):
                        entities = perform_filtered_ner(user_input)
                    if entities:
                        st.success("✅ Named Entities detected:")
                        for ent in entities:
                            st.markdown(f"{ent['text']} ({ent['label']})")
                    else:
                        st.info("ℹ️ No relevant entities detected.")

                    # Load Trained Model and Label Encoder
                    with st.spinner("📦 Loading transformer model and label encoder..."):
                        propaganda_model, propaganda_tokenizer, propaganda_mlb = load_transformer_model(
                            DEFAULT_CONFIG['propaganda_model_file'],
                            DEFAULT_CONFIG['label_encoder_file']
                        )
                    st.success("✅ Loaded transformer model and label encoder.")

                    # Predict Propaganda Techniques
                    with st.spinner("🔍 Predicting propaganda techniques..."):
                        predicted_techniques = predict_with_transformer(
                            propaganda_model,
                            propaganda_tokenizer,
                            propaganda_mlb,
                            [processed_text],
                            threshold=0.5
                        )[0]
                    if predicted_techniques:
                        st.success("✅ Predicted Propaganda Techniques:")
                        st.markdown(", ".join(predicted_techniques))
                    else:
                        st.info("ℹ️ No propaganda techniques predicted.")

                    # Load Media Bias Detection Model
                    with st.spinner("🔍 Loading media bias detection model..."):
                        bias_model, bias_tokenizer = load_bias_model(DEFAULT_CONFIG['bias_model_file'])
                    st.success("✅ Loaded media bias detection model.")

                    # Load or Define MultiLabelBinarizer for Media Bias Categories
                    bias_mlb_file = DEFAULT_CONFIG['media_bias_label_encoder_file']
                    bias_mlb = load_media_bias_label_encoder(bias_mlb_file)

                    # Predict Media Bias
                    with st.spinner("🔍 Predicting media bias..."):
                        bias_category = detect_media_bias(user_input, bias_model, bias_tokenizer, bias_mlb, threshold=0.5)
                    st.success(f"✅ Predicted Media Bias Category: **{bias_category}**")

                except Exception as e:
                    logging.error(f"An error occurred during text analysis: {type(e).__name__} - {e}")
                    st.error(f"An error occurred during text analysis: {type(e).__name__} - {e}")
            else:
                st.warning("⚠️ Please enter some text to analyze.")

if __name__ == '__main__':
    main()
