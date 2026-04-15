import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv")

# Train model
X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X, y)

# UI
st.title("📧 Email Spam Detector")

email = st.text_area("Enter Email Content:")

if st.button("Check Spam"):
    if email:
        email_vec = vectorizer.transform([email])
        prediction = model.predict(email_vec)[0]

        if prediction == "spam":
            st.error("🚫 This is SPAM")
        else:
            st.success("✅ This is NOT Spam")
    else:
        st.warning("Please enter email text")