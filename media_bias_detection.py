# --- Load SpaCy Model ---
SPACY_MODEL = "en_core_web_sm"

try:
    # Attempt to load the spaCy model
    nlp = spacy.load(SPACY_MODEL)
    logger.info(f"SpaCy model '{SPACY_MODEL}' loaded successfully.")
except OSError:
    logger.warning(f"SpaCy model '{SPACY_MODEL}' not found. Attempting to download...")
    try:
        # Try downloading the model if not present
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL], check=True)
        nlp = spacy.load(SPACY_MODEL)
        logger.info(f"SpaCy model '{SPACY_MODEL}' downloaded and loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download and load SpaCy model '{SPACY_MODEL}': {e}")
        st.error(f"Failed to download and load the SpaCy model '{SPACY_MODEL}'. Please ensure that the model is available and compatible.")
        st.stop()
except ImportError as e:
    logger.error(f"ImportError: {e}")
    st.error("An ImportError occurred. Please ensure all required packages are installed correctly.")
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
    # Default bias terms
    bias_terms = [
        'always', 'never', 'obviously', 'clearly', 'undoubtedly', 'unquestionably',
        'everyone knows', 'no one believes', 'definitely', 'certainly', 'extremely',
        'inconceivable', 'must', 'prove', 'disprove', 'true', 'false'
    ]
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

# --- Analysis Functions ---

def perform_analysis(text, title="Article"):
    if not text:
        st.error("No text to analyze.")
        return None

    models = initialize_models()
    nlp = models['nlp']
    sentiment_pipeline = models['sentiment_pipeline']
    propaganda_pipeline = models['propaganda_pipeline']

    # Preprocess Text
    text = preprocess_text(text)

    # Split text into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    total_sentences = len(sentences)

    if total_sentences == 0:
        st.error("The article has no valid sentences to analyze.")
        return None

    # Sentiment Analysis
    try:
        sentiment_results = sentiment_pipeline(sentences, batch_size=4, truncation=True)
        sentiment_scores = []
        for result in sentiment_results:
            label = result['label']
            # Map labels to numerical scores
            if label == '1 star':
                sentiment_scores.append(1)
            elif label == '2 stars':
                sentiment_scores.append(2)
            elif label == '3 stars':
                sentiment_scores.append(3)
            elif label == '4 stars':
                sentiment_scores.append(4)
            elif label == '5 stars':
                sentiment_scores.append(5)
            else:
                sentiment_scores.append(3)  # Neutral
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 3
        avg_sentiment_score = round(avg_sentiment_score, 2)
        # Interpret the average sentiment score
        if avg_sentiment_score >= 3.5:
            sentiment_label = 'Positive'
        elif avg_sentiment_score <= 2.5:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        avg_sentiment_score = 3
        sentiment_label = 'Neutral'
        logger.error(f"Error during sentiment analysis: {e}")

    # Bias Detection
    biased_sentences = []
    bias_terms = st.session_state.get('bias_terms', load_default_bias_terms())
    bias_terms = [term.lower() for term in bias_terms]
    for sentence in sentences:
        doc_sentence = nlp(sentence)
        sentence_tokens = set([token.text.lower() for token in doc_sentence])
        detected_terms = list(set([term for term in bias_terms if term in sentence_tokens]))
        if detected_terms:
            biased_sentences.append({'sentence': sentence, 'detected_terms': detected_terms})

    bias_count = len(biased_sentences)

    # Fix Bias Terms Duplication
    for item in biased_sentences:
        item['detected_terms'] = list(set(item['detected_terms']))

    # Propaganda Detection
    propaganda_sentences = []
    try:
        for i in range(0, len(sentences), 8):  # Batch size of 8 for efficiency
            batch_sentences = sentences[i:i+8]
            predictions = propaganda_pipeline(batch_sentences)
            for idx, prediction in enumerate(predictions):
                label = prediction['label']
                score = prediction['score']
                if label.lower() == 'propaganda' and score >= 0.9:
                    propaganda_sentences.append({
                        'sentence': batch_sentences[idx],
                        'score': score,
                        'label': label
                    })
        propaganda_count = len(propaganda_sentences)
    except Exception as e:
        st.error(f"Error during propaganda detection: {e}")
        logger.error(f"Error during propaganda detection: {e}")
        propaganda_count = 0

    # Final Score Calculation
    sentiment_percentage = ((avg_sentiment_score - 1) / 4) * 100  # Scale from 1-5 to 0-100%
    # Subtract penalties proportional to the total number of sentences
    bias_penalty = (bias_count / total_sentences) * 50  # Max penalty of 50
    propaganda_penalty = (propaganda_count / total_sentences) * 50  # Max penalty of 50
    final_score = sentiment_percentage - bias_penalty - propaganda_penalty
    final_score = max(0, min(100, final_score))  # Ensure score is between 0 and 100

    # Save analysis data
    analysis_data = {
        'title': title,
        'text': text,
        'sentiment_score': avg_sentiment_score,
        'sentiment_label': sentiment_label,
        'bias_score': bias_count,
        'biased_sentences': biased_sentences,
        'propaganda_score': propaganda_count,
        'propaganda_sentences': propaganda_sentences,
        'final_score': final_score,
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'email': st.session_state.get('email', 'guest')
    }

    return analysis_data

# --- Display Functions ---

def display_results(data, is_nested=False):
    with st.container():
        st.markdown(f"## {data.get('title', 'Untitled Article')}")
        st.markdown(f"**Date:** {data.get('date', 'N/A')}")
        st.markdown(f"**Analyzed by:** {data.get('email', 'guest')}")

        # Overview Metrics in Columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sentiment_label = data.get('sentiment_label', 'Neutral')
            sentiment_score = data.get('sentiment_score', 3.0)
            if sentiment_label == "Positive":
                sentiment_color = "#28a745"  # Green
            elif sentiment_label == "Negative":
                sentiment_color = "#dc3545"  # Red
            else:
                sentiment_color = "#6c757d"  # Gray
            st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};'>{sentiment_label}</span>", unsafe_allow_html=True)
            st.metric(label="Sentiment Score", value=f"{sentiment_score:.2f}")

        with col2:
            bias_count = data.get('bias_score', 0)
            st.markdown("**Bias Count**")
            st.metric(label="Bias Terms Detected", value=f"{int(bias_count)}")

        with col3:
            st.markdown("**Propaganda Count**")
            propaganda_count = data.get('propaganda_score', 0)
            st.metric(label="Propaganda Sentences Detected", value=f"{int(propaganda_count)}")

        with col4:
            final_score = data.get('final_score', 0.0)
            st.markdown("**Final Score**")
            st.metric(
                label="Final Score",
                value=f"{final_score:.2f}",
                help="A score out of 100 indicating overall article quality."
            )

        st.markdown("---")  # Separator

        # Tabs for Different Analysis Sections
        tabs = st.tabs(["Sentiment Analysis", "Bias Detection", "Propaganda Detection"])

        # --- Sentiment Analysis Tab ---
        with tabs[0]:
            st.markdown("### Sentiment Analysis")
            st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{sentiment_label}</span>", unsafe_allow_html=True)
            st.write(f"**Average Sentiment Score:** {sentiment_score:.2f} out of 5")

        # --- Bias Detection Tab ---
        with tabs[1]:
            st.markdown("### Bias Detection")
            st.write(f"**Bias Count:** {int(bias_count)} bias terms detected.")

            if data.get('biased_sentences'):
                for idx, item in enumerate(data['biased_sentences'], 1):
                    st.markdown(f"**{idx}. Sentence:** {item['sentence']}")
                    st.markdown(f"   - **Detected Bias Terms:** {', '.join(set(item['detected_terms']))}")
            else:
                st.write("No biased sentences detected.")

        # --- Propaganda Detection Tab ---
        with tabs[2]:
            st.markdown("### Propaganda Detection")
            st.write(f"**Propaganda Count:** {int(propaganda_count)} propaganda sentences detected.")

            if data.get('propaganda_sentences'):
                for idx, item in enumerate(data['propaganda_sentences'], 1):
                    st.markdown(f"**{idx}. Sentence:** {item['sentence']}")
                    st.markdown(f"   - **Detected Label:** {item['label']}")
                    st.markdown(f"   - **Confidence Score:** {item['score']:.2f}")
            else:
                st.write("No propaganda detected.")

        st.markdown("---")  # Separator

        # Save to history if logged in
        if st.session_state.get('logged_in', False) and not is_nested:
            save_analysis_to_history(data)
            st.success("Analysis saved to your history.")

            # Provide Download Option for CSV
            csv_data = {
                'title': data.get('title', 'Untitled'),
                'date': data.get('date', ''),
                'sentiment_score': data.get('sentiment_score', 3.0),
                'sentiment_label': data.get('sentiment_label', 'Neutral'),
                'bias_count': data.get('bias_score', 0),
                'propaganda_count': data.get('propaganda_score', 0),
                'final_score': data.get('final_score', 0.0),
            }
            df_csv = pd.DataFrame([csv_data])
            csv_buffer = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Analysis as CSV",
                data=csv_buffer,
                file_name=f"analysis_{data.get('title', 'untitled').replace(' ', '_')}.csv",
                mime='text/csv'
            )

        # --- Feedback Section ---
        if not is_nested:
            st.markdown("---")
            st.markdown("### Provide Feedback")
            feedback = st.text_area(
                "Your Feedback",
                placeholder="Enter your feedback here...",
                height=100
            )
            if st.button("Submit Feedback"):
                if feedback:
                    # Save feedback to a JSON file
                    feedback_path = 'feedback.json'
                    feedback_entry = {
                        'email': st.session_state.get('email', 'guest'),
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'feedback': feedback
                    }
                    try:
                        if not os.path.exists(feedback_path):
                            with open(feedback_path, 'w') as f:
                                json.dump([], f)
                            logger.info("Created new feedback file.")
                        with open(feedback_path, 'r') as f:
                            feedback_data = json.load(f)
                        feedback_data.append(feedback_entry)
                        with open(feedback_path, 'w') as f:
                            json.dump(feedback_data, f, indent=4)
                        logger.info(f"Feedback saved from user '{st.session_state.get('email', 'guest')}'.")
                        st.success("Thank you for your feedback!")
                    except Exception as e:
                        logger.error(f"Error saving feedback: {e}", exc_info=True)
                        st.error("An error occurred while saving your feedback.")
                else:
                    st.warning("Please enter your feedback before submitting.")

# --- Single Article Analysis Function ---
def single_article_analysis():
    st.header("Single Article Analysis")
    st.write("Enter the article URL or paste the article text below.")

    input_type = st.radio(
        "Select Input Type",
        ['Enter URL', 'Paste Article Text'],
        key="single_article_input_type"
    )
    if input_type == 'Enter URL':
        url = st.text_input(
            "Article URL",
            placeholder="https://example.com/article",
            key="single_article_url"
        ).strip()
        article_text = ''
    else:
        article_text = st.text_area(
            "Article Text",
            placeholder="Paste the article text here...",
            height=300,
            key="single_article_text"
        ).strip()
        url = ''

    title = st.text_input(
        "Article Title",
        value="Article",
        placeholder="Enter a title for the article",
        key="single_article_title"
    )

    if st.button("Analyze", key="analyze_single_article"):
        if input_type == 'Enter URL':
            if url:
                if is_valid_url(url):
                    with st.spinner('Fetching the article...'):
                        article_text_fetched = fetch_article_text(url)
                        if article_text_fetched:
                            sanitized_text = preprocess_text(article_text_fetched)
                            st.success("Article text fetched successfully.")
                            article_text = sanitized_text
                        else:
                            st.error("Failed to fetch article text. Please check the URL and try again.")
                            return
                else:
                    st.error("Please enter a valid URL.")
                    return
            else:
                st.error("Please enter a URL.")
                return
        else:
            if not article_text.strip():
                st.error("Please paste the article text.")
                return
            article_text = preprocess_text(article_text)

        with st.spinner('Performing analysis...'):
            data = perform_analysis(article_text, title)
            if data:
                if st.session_state.get('logged_in', False):
                    data['email'] = st.session_state['email']
                else:
                    data['email'] = 'guest'
                st.success("Analysis completed successfully.")
                display_results(data)
            else:
                st.error("Failed to perform analysis on the provided article.")

# --- Comparative Analysis Function ---
def comparative_analysis():
    st.header("Comparative Analysis")
    st.write("Compare multiple articles side by side.")

    num_articles = st.number_input("Number of articles to compare", min_value=2, max_value=5, value=2, step=1)

    article_texts = []
    titles = []
    analyses = []

    for i in range(int(num_articles)):
        st.subheader(f"Article {i+1}")
        input_type = st.radio(
            f"Select Input Type for Article {i+1}",
            ['Enter URL', 'Paste Article Text'],
            key=f"comp_input_type_{i}"
        )
        if input_type == 'Enter URL':
            url = st.text_input(
                f"Article URL for Article {i+1}",
                placeholder="https://example.com/article",
                key=f"comp_url_{i}"
            ).strip()
            article_text = ''
            if url:
                if is_valid_url(url):
                    with st.spinner(f'Fetching the article {i+1}...'):
                        article_text_fetched = fetch_article_text(url)
                        if article_text_fetched:
                            sanitized_text = preprocess_text(article_text_fetched)
                            st.success(f"Article {i+1} text fetched successfully.")
                            article_text = sanitized_text
                        else:
                            st.error(f"Failed to fetch article text for Article {i+1}. Please check the URL and try again.")
                            return
                else:
                    st.error(f"Please enter a valid URL for Article {i+1}.")
                    return
        else:
            article_text = st.text_area(
                f"Article Text for Article {i+1}",
                height=200,
                key=f"comp_text_{i}"
            )
            if not article_text.strip():
                st.error(f"Please paste the article text for Article {i+1}.")
                return
            article_text = preprocess_text(article_text)
        title = st.text_input(
            f"Title for Article {i+1}",
            key=f"comp_title_{i}"
        )
        titles.append(title)
        article_texts.append(article_text)

    if st.button("Analyze Articles", key="compare_articles"):
        for i, text in enumerate(article_texts):
            if text.strip():
                with st.spinner(f"Analyzing Article {i+1}..."):
                    data = perform_analysis(text, titles[i])
                    if data:
                        analyses.append(data)
            else:
                st.error(f"Please provide text for Article {i+1}.")
                return
        if analyses:
            st.success("Comparative Analysis Completed.")
            display_comparative_results(analyses)
        else:
            st.error("Failed to perform comparative analysis.")

def display_comparative_results(analyses):
    st.markdown("## Comparative Results")

    df = pd.DataFrame([
        {
            'Title': data['title'],
            'Sentiment Score': data['sentiment_score'],
            'Sentiment Label': data['sentiment_label'],
            'Bias Count': data['bias_score'],
            'Propaganda Count': data['propaganda_score'],
            'Final Score': data['final_score'],
        }
        for data in analyses
    ])

    st.dataframe(df.style.highlight_max(axis=0))

    # Detailed Results
    for data in analyses:
        st.markdown("---")
        display_results(data, is_nested=True)

# --- Settings Page Function ---
def settings_page():
    st.header("Settings")
    st.write("Customize your analysis settings.")

    # Manage Bias Terms
    st.subheader("Manage Bias Terms")

    # Input field to add a new bias term
    with st.form("add_bias_term_form"):
        new_bias_term = st.text_input(
            "Add a New Bias Term",
            placeholder="Enter new bias term",
            key="add_bias_term_input"
        )
        submitted = st.form_submit_button("Add Term")
        if submitted:
            if new_bias_term:
                if new_bias_term.lower() in [term.lower() for term in st.session_state['bias_terms']]:
                    st.warning("This bias term already exists.")
                else:
                    st.session_state['bias_terms'].append(new_bias_term)
                    st.success(f"Added new bias term: {new_bias_term}")
                    logger.info(f"Added new bias term: {new_bias_term}")
                # Remove duplicates
                st.session_state['bias_terms'] = list(set(st.session_state['bias_terms']))
            else:
                st.warning("Please enter a valid bias term.")

    # Text area to edit bias terms
    st.subheader("Edit Bias Terms")
    bias_terms_str = '\n'.join(st.session_state['bias_terms'])
    edited_bias_terms_str = st.text_area(
        "Edit Bias Terms (one per line)",
        value=bias_terms_str,
        height=200,
        key="edit_bias_terms_textarea"
    )
    if st.button("Save Bias Terms", key="save_bias_terms_button"):
        updated_bias_terms = [term.strip() for term in edited_bias_terms_str.strip().split('\n') if term.strip()]
        # Remove duplicates and ensure uniqueness
        unique_terms = []
        seen = set()
        for term in updated_bias_terms:
            lower_term = term.lower()
            if lower_term not in seen:
                unique_terms.append(term)
                seen.add(lower_term)
        st.session_state['bias_terms'] = unique_terms
        st.success("Bias terms updated successfully.")
        logger.info("Updated bias terms list.")

    # Button to reset bias terms to default
    if st.button("Reset Bias Terms to Default", key="reset_bias_terms"):
        st.session_state['bias_terms'] = load_default_bias_terms()
        st.success("Bias terms have been reset to default.")
        logger.info("Reset bias terms list.")

    st.markdown("### Note:")
    st.markdown("Use the **'Add a New Bias Term'** form to introduce new terms. You can edit existing terms in the text area above. To reset to the default bias terms, click the **'Reset Bias Terms to Default'** button.")

# --- Help Page Function ---
def help_feature():
    st.header("Help")
    st.write("""
    **Media Bias Detection Tool** helps you analyze articles for sentiment, bias, and propaganda. Here's how to use the tool:

    ### **1. Single Article Analysis**
    - **Input Type**: Choose to either enter a URL or paste the article text directly.
    - **Article Title**: Provide a title for your reference.
    - **Analyze**: Click the "Analyze" button to perform the analysis.
    - **Save Analysis**: After analysis, it is automatically saved to your history.

    ### **2. Comparative Analysis**
    - **Compare Articles**: Compare multiple articles side by side.
    - **Input Articles**: Provide titles and texts or URLs for each article.
    - **Analyze**: Click "Analyze Articles" to perform comparative analysis.

    ### **3. Settings**
    - **Manage Bias Terms**: Add new bias terms or edit existing ones to customize the analysis.
    - **Reset Terms**: Revert to the default bias terms if needed.

    ### **4. Login & Registration**
    - **Register**: Create a new account with a unique email and strong password.
    - **Login**: Access your personalized dashboard with your preferences and history.
    - **Reset Password**: If you forget your password, use the reset password feature.
    - **Logout**: Securely log out of your account.

    ### **5. Feedback**
    - **Provide Feedback**: After analysis, you can provide feedback to help improve the tool.

    ### **6. Download Analysis**
    - **Download as CSV**: After analysis, download your results as a CSV file for your records.

    If you encounter any issues or have questions, please refer to the documentation or contact support.
    """)

# --- History Page Function ---
def display_history():
    st.header("Your Analysis History")
    email = st.session_state.get('email', '')
    history = load_user_history(email)

    if not history:
        st.info("No history available.")
        return

    # Convert history to DataFrame for sorting
    history_df = pd.DataFrame(history)

    # Sort history by date descending
    history_df['date'] = pd.to_datetime(history_df['date'])
    history_df_sorted = history_df.sort_values(by='date', ascending=False)

    for idx, entry in history_df_sorted.iterrows():
        with st.expander(f"{entry.get('title', 'Untitled')} - {entry.get('date', 'N/A')}", expanded=False):
            entry_dict = entry.to_dict()
            display_results(entry_dict, is_nested=True)

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

# --- Main Function ---

def main():
    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'email' not in st.session_state:
        st.session_state['email'] = ''
    if 'bias_terms' not in st.session_state:
        st.session_state['bias_terms'] = load_default_bias_terms()

    # Display Python version (for debugging purposes)
    python_version = sys.version.split(' ')[0]
    st.sidebar.markdown(f"**Python Version:** {python_version}")

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
