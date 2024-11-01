# media_bias_detection.py

import streamlit as st
import logging
from transformers import pipeline
import spacy

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the spaCy model directly by name
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("SpaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SpaCy model: {e}")
    st.error("Failed to load SpaCy model. Ensure it is installed correctly via pip.")
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
