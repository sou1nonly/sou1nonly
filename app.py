import streamlit as st
import joblib
import string
import spacy
import subprocess
import sys
import os

def install(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Path to the wheel file
wheel_file = os.path.join(current_directory, "en_core_web_sm-3.5.0-py3-none-any.whl")

# Install the package if it is not already installed
try:
    import spacy
except ImportError:
    print("Spacy is not installed. Installing...")
    install("spacy")

# Check if the model is already installed
try:
    nlp = spacy.load("en_core_web_sm")
    print("Model 'en_core_web_sm' is already installed.")
except OSError:
    print("Model 'en_core_web_sm' is not installed. Installing from wheel file...")
    install(wheel_file)
    nlp = spacy.load("en_core_web_sm")

# Example text to process
text = "Hello, world! This is a test sentence."

# Process the text
doc = nlp(text)

# Print tokens and their part-of-speech tags
for token in doc:
    print(token.text, token.pos_)

st.set_page_config(page_title="Spam Email Detector", page_icon="✉️")

@st.cache_resource(ttl=3600)
def load_model():
    model = joblib.load('spam_detection_model.pkl')
    vectorizer = joblib.load('count_vectorizer.pkl')
    return model, vectorizer

def clean_text(text):
    return ''.join(char if char in string.ascii_letters + ' ' else ' ' for char in text).strip()

def preprocess(text):
    cleaned_text = clean_text(text)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(cleaned_text)
    return " ".join(token.lemma_ for token in doc if len(token) > 2)

def classify_email(model, vectorizer, email):
    return model.predict(vectorizer.transform([email]))

def main():
    st.title("Spam Email Detector")
    user_input = st.text_input('Enter the email text:', '', placeholder='you have a important meeting coming up...')

    if st.button("Check for Spam"):
        if user_input:
            model, vectorizer = load_model()
            processed_input = preprocess(user_input)
            st.write("Processed Input:", processed_input)  

            prediction = classify_email(model, vectorizer, processed_input)  

            if prediction == 1:
                st.error('Spam Detected!')
            else:
                st.success('Not Spam')
        else:
            st.warning("Please enter the text to detect!")

if __name__ == "__main__":
    main()
