import streamlit as st
import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import speech_recognition as sr
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load models
nlp = spacy.load("en_core_web_sm")
model = load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

analyzer = SentimentIntensityAnalyzer()

# Helper functions
def sentiment_with_emoji(sentiment):
    return {
        "Positive": "Positive 😊",
        "Negative": "Negative 😠",
        "Neutral": "Neutral 😐"
    }.get(sentiment, sentiment)

def predict_sentiment(sentence, max_len=100):
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

# def get_aspect_phrase(aspect_token):
#     subtree = list(aspect_token.subtree)
#     start = min([token.i for token in subtree])
#     end = max([token.i for token in subtree]) + 1
#     return aspect_token.doc[start:end].text

def get_aspect_phrase(aspect_token):
    """
    Hybrid approach to extract an aspect phrase:
    1. Uses syntactic dependency to get opinions.
    2. Falls back to punctuation-based trimming if needed.
    3. Returns natural-sounding short phrase focused on the aspect.
    """
    aspect = aspect_token.text
    head = aspect_token.head

    # Rule 1: Aspect is subject to an opinion adjective
    if aspect_token.dep_ == "nsubj" and head.pos_ == "ADJ":
        return f"The {aspect} is {head.text}."

    # Rule 2: Aspect has adjective child (e.g., 'battery' → 'disappointing')
    for child in aspect_token.children:
        if child.pos_ == "ADJ":
            return f"The {aspect} is {child.text}."

    # Rule 3: Aspect is part of a noun phrase and sentence has sentiment clue
    phrase = aspect_token.sent.text
    start = phrase.find(aspect)
    if start != -1:
        trimmed = phrase[start:]
        for stop_char in [',', '.', '!', '?']:
            if stop_char in trimmed:
                trimmed = trimmed.split(stop_char, 1)[0] + stop_char
                break
        return f"The {trimmed}".strip()

    # Rule 4: Fallback — return sentence
    return aspect_token.sent.text.strip()


# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def sentiment_with_emoji(sentiment):
    """
    Maps sentiment label to an emoji.
    """
    return {
        "Positive": "Positive 😊",
        "Negative": "Negative 😞",
        "Neutral": "Neutral 😐"
    }.get(sentiment, "❓")

def analyze_aspects(sentence):
    """
    Performs aspect-based sentiment analysis:
    - Extracts nouns (aspects)
    - Generates trimmed phrase per aspect
    - Analyzes sentiment using VADER
    - Returns a DataFrame with aspect, phrase, sentiment, and emoji
    """
    doc = nlp(sentence)

    # Step 1: Extract unique noun/proper noun tokens as aspects
    aspects = list({token for token in doc if token.pos_ in ["NOUN", "PROPN"]})

    # Step 2: Analyze each aspect's sentiment
    results = []
    for aspect in aspects:
        phrase = get_aspect_phrase(aspect)
        score = analyzer.polarity_scores(phrase)['compound']
        if score > 0.05:
            sentiment = "Positive"
        elif score < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        results.append({
            "Aspect": aspect.text,
            "Phrase": phrase,
            "Sentiment": sentiment,
            "Sentiment with Emoji": sentiment_with_emoji(sentiment)
        })

    return pd.DataFrame(results)

def plot_sentiment_pie(df):
    counts = df["Sentiment"].value_counts()
    labels = counts.index.tolist()
    sizes = counts.values.tolist()
    colors = ["#8BC34A" if label == "Positive" else "#F44336" if label == "Negative" else "#FFC107" for label in labels]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    return fig

def plot_aspect_bar(df):
    fig, ax = plt.subplots()
    colors = df["Sentiment"].map({"Positive": "#8BC34A", "Neutral": "#FFC107", "Negative": "#F44336"})
    ax.bar(df["Aspect"], [1]*len(df), color=colors)
    ax.set_title("Aspect Sentiment (Color-Coded)")
    ax.set_ylabel("Sentiment")
    ax.set_yticks([])
    return fig

def capture_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak clearly.")
        audio = r.listen(source, timeout=5)
        try:
            text = r.recognize_google(audio)
            st.success(f"🗣️ You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your voice.")
        except sr.RequestError:
            st.error("Could not request results; check your internet connection.")
    return ""

# --- UI ---
st.set_page_config(page_title="Live Sentiment Analyzer with Voice", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4ECDC4;'>🧠  Aspect Based Sentiment Analysis based on movie reviews using LSTM</h1>", unsafe_allow_html=True)

# Voice input
if st.button("🎙️ Speak Now"):
    spoken_text = capture_voice()
    st.session_state['user_sentence'] = spoken_text

# User input
user_input = st.text_area("💬 Type or dictate your sentence:", value=st.session_state.get('user_sentence', ""), height=120)

# Analyze Button
if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter a sentence.")
    else:
        # --- Sentiment Prediction ---
        # overall_sentiment = predict_sentiment(user_input)
        # aspect_df = analyze_aspects(user_input)
        score = analyzer.polarity_scores(user_input)['compound']
        # overall_sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        overall_sentiment = predict_sentiment(user_input)
        aspect_df = analyze_aspects(user_input)

        # --- Overall Sentiment Output with background color ---
        st.markdown("## 💬 Overall Sentiment")
        st.markdown(f"**Sentence:** {user_input}")

        sentiment_color = {
            "Positive": "#75c72e",  # light green
            "Neutral": "#d4b924",   # light yellow
            "Negative": "#db582c"   # light red
        }.get(overall_sentiment, "#ffffff")

        sentiment_emoji = {
            "Positive": "😊",
            "Neutral": "😐",
            "Negative": "😠"
        }.get(overall_sentiment, "")

        st.markdown(
            f"""
            <div style="background-color: {sentiment_color}; padding: 15px; border-radius: 8px; font-size: 20px; text-align: center;">
                <b>Prediction:</b> {sentiment_emoji} <b>{overall_sentiment}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("📂 Aspect-Based Breakdown")
        df_aspects = analyze_aspects(user_input)

        if df_aspects.empty:
                st.info("No aspects (nouns/proper nouns) found.")
        else:
                selected_aspect = st.selectbox("🔎 Filter by aspect (optional):", options=["All"] + list(df_aspects["Aspect"]), index=0)
                filtered_df = df_aspects if selected_aspect == "All" else df_aspects[df_aspects["Aspect"] == selected_aspect]

                st.dataframe(filtered_df[["Aspect", "Phrase", "Sentiment with Emoji"]], use_container_width=True)

                chart1, chart2 = st.columns(2)
        

        # --- Pie chart and Aspect Bar Chart side by side ---
        st.markdown("## 📊 Aspect-Based Sentiment Visuals")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Sentiment Pie")
            st.pyplot(plot_sentiment_pie(aspect_df))

        with col2:
            st.subheader("📊 Aspect Bar Chart")
            fig_aspect, ax_aspect = plt.subplots(figsize=(5, max(3, len(aspect_df)*0.5)))
            aspect_colors = aspect_df["Sentiment"].map({
                "Positive": "#8BC34A", "Neutral": "#FFC107", "Negative": "#F44336"
            })
            ax_aspect.barh(aspect_df["Aspect"], 1, color=aspect_colors)
            ax_aspect.set_yticks(range(len(aspect_df)))
            ax_aspect.set_yticklabels(aspect_df["Aspect"])
            ax_aspect.set_xticks([])
            ax_aspect.set_xlim(0, 1.5)
            ax_aspect.set_title("Aspect Sentiment (Color-Coded)")
            st.pyplot(fig_aspect)

        # --- Center-align Export Button ---
            st.subheader("📁 Export Results")
            csv_buffer = BytesIO()
            aspect_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download CSV", data=csv_buffer.getvalue(), file_name="aspect_sentiment.csv", mime="text/csv")


    st.markdown("---")
    st.markdown("Developed with ❤...", unsafe_allow_html=True)
