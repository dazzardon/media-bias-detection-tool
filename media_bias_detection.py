# media_bias_detection.py

import streamlit as st
import logging
import os
from pathlib import Path
import tarfile
import spacy

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Paths
MODEL_TAR_PATH = Path("/mount/src/media-bias-detection-tool/models/en_core_web_sm-3.5.0.tar.gz")
EXTRACTED_MODEL_DIR = Path("/mount/src/media-bias-detection-tool/models/en_core_web_sm")

# Model Loading and Extraction
try:
    # Check if the extracted model directory exists
    if not EXTRACTED_MODEL_DIR.exists():
        with tarfile.open(MODEL_TAR_PATH, "r:gz") as tar:
            tar.extractall(path=EXTRACTED_MODEL_DIR.parent)  # Extract tar.gz into the models directory
            logger.info(f"Extracted spaCy model to {EXTRACTED_MODEL_DIR.parent}")

        # Check if the extracted folder has the version info in its name
        extracted_dir = EXTRACTED_MODEL_DIR.parent / "en_core_web_sm-3.5.0"
        if extracted_dir.exists():
            # Rename the directory to match spaCy's loading requirements
            os.rename(extracted_dir, EXTRACTED_MODEL_DIR)
            logger.info(f"Renamed extracted model directory to {EXTRACTED_MODEL_DIR}")

    # Verify files in extracted directory and check for 'config.cfg'
    contents = list(EXTRACTED_MODEL_DIR.rglob("*"))
    logger.info(f"Contents of {EXTRACTED_MODEL_DIR}: {[str(item) for item in contents]}")
    if not (EXTRACTED_MODEL_DIR / "config.cfg").exists():
        raise FileNotFoundError("config.cfg not found in extracted model directory.")

    # Load the spaCy model
    nlp = spacy.load(EXTRACTED_MODEL_DIR)
    logger.info("SpaCy model loaded successfully.")

except Exception as e:
    logger.error(f"Failed to load SpaCy model from '{MODEL_TAR_PATH}': {e}")
    st.error("Failed to load SpaCy model. Check model placement and compatibility.")
    st.stop()

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
    # Use the already loaded nlp model
    models = {
        'sentiment_pipeline': sentiment_pipeline_model,
        'propaganda_pipeline': propaganda_pipeline_model,
        'nlp': nlp
    }
    return models

