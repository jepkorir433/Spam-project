import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    df['cleaned'] = df['message'].apply(clean_text)
    return df

# Train model
@st.cache_resource
def train_model(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['label_num']
    model = MultinomialNB()
    model.fit(X, y)
    return model, vectorizer

# Predict
def predict_message(message, model, vectorizer):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "🚫 Spam" if prediction[0] == 1 else "✅ Not Spam", prediction[0]

# App UI
st.set_page_config(page_title="Spam Email Classifier", page_icon="📩")
st.title("📩 Spam Email Classifier")
st.markdown("Check if a message is **spam or not** using a machine learning model.")

st.divider()

df = load_data()
model, vectorizer = train_model(df)

# User input
user_input = st.text_area("✉️ Enter your message here:", height=150)
if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        result, pred_value = predict_message(user_input, model, vectorizer)
        st.success(f"**Result:** {result}")

        # Chart update
        fig, ax = plt.subplots()
        labels = ['Not Spam', 'Spam']
        sizes = [1 - pred_value, pred_value]
        colors = ['#00cc99', '#ff6666']
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.pyplot(fig)

# Optional: file uploader
st.divider()
st.subheader("📁 Optional: Upload a File to Check Multiple Messages")
uploaded_file = st.file_uploader("Upload a .txt file (one message per line)", type=["txt"])
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    lines = content.strip().split('\n')
    predictions = []
    for line in lines:
        _, pred = predict_message(line, model, vectorizer)
        predictions.append(pred)

    results_df = pd.DataFrame({
        'Message': lines,
        'Prediction': ['Spam' if p == 1 else 'Not Spam' for p in predictions]
    })
    st.write(results_df)

    # Chart
    fig2, ax2 = plt.subplots()
    counts = results_df['Prediction'].value_counts()
    ax2.bar(counts.index, counts.values, color=['#00cc99', '#ff6666'])
    ax2.set_title("Spam vs Not Spam")
    st.pyplot(fig2)

