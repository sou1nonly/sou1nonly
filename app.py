import streamlit as st
import joblib
import string
import time

st.set_page_config(page_title="Spam Email Detector", page_icon="✉️")

model = joblib.load('spam_detection_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c in string.ascii_letters or c == ' ')
    return text

def classify_email(model, vectorizer, email):
    cleaned_email = clean_text(email)
    vectorized_email = vectorizer.transform([cleaned_email])
    prediction = model.predict(vectorized_email)[0]
    probability = model.predict_proba(vectorized_email)[0][1] * 100
    return prediction, probability

def main():
    st.title("Spam Email Detector")
    user_input = st.text_input('Enter the email text:', '', placeholder='you have a important meeting coming up...')

    if st.button("Check for Spam"):
        if user_input:
            with st.spinner('Processing...'):
                # Simulate processing time (adjust as needed)
                time.sleep(2)

                prediction, probability = classify_email(model, vectorizer, user_input)

                if prediction == 1:
                    st.error('Spam Detected!')
                    st.write(f"Probability of being spam: {probability:.2f}%")
                else:
                    st.success('Not Spam')
                    st.write(f"Probability of being spam: {probability:.2f}%")
        else:
            st.warning("Please enter the email text.")

if __name__ == "__main__":
    main()
