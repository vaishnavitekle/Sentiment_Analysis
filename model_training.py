import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load and preprocess dataset
df = pd.read_csv("IMDB Dataset.csv")

# Drop rows where sentiment is NaN (caused by unexpected values)
df.dropna(subset=['sentiment', 'review'], inplace=True)

def clean_text(text):
    text = text.lower()                             # Lowercase
    text = re.sub(r"<.*?>", "", text)               # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)         # Remove punctuation/numbers
    text = text.split()                             # Tokenize
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    return " ".join(text)

df['cleaned_review'] = df['review'].apply(clean_text)

# Label Encoding: Convert sentiment into numerical values
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['sentiment'])

# Tokenization and Padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_review'])
sequences = tokenizer.texts_to_sequences(df['cleaned_review'])

# Pad sequences
X = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
y = df['label'].values
# Reset index after drop
df.reset_index(drop=True, inplace=True)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])

X = tokenizer.texts_to_sequences(df['review'])
X = pad_sequences(X, maxlen=100, padding='post', truncating='post')
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(10000, 128, input_length=100),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Save model and tokenizer
model.save("sentiment_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)



# --- Predictions and Thresholding ---
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# --- Calculate Metrics ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Plot Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# --- Plot Bar Chart of Metrics ---
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

plt.figure(figsize=(6, 4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='Set2')
plt.ylim(0, 1.05)
plt.title('Performance Metrics')
plt.ylabel('Score')
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()