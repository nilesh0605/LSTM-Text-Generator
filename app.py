import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ⚠️ same max_len as training (approx)
max_len = 20

st.title("🔥 LSTM Text Generator")

seed = st.text_input("Enter seed text")

def generate_text(seed_text, next_words=15):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        pred_probs = model.predict(token_list, verbose=0)[0]
        # predicted = np.random.choice(len(pred_probs), p=pred_probs)
        predicted = np.random.choice(len(pred_probs), p=pred_probs / np.sum(pred_probs))
        # 🔥 Temperature control
        temperature = 0.6

        pred_probs = np.log(pred_probs + 1e-8) / temperature
        pred_probs = np.exp(pred_probs)
        pred_probs = pred_probs / np.sum(pred_probs)

        predicted = np.random.choice(len(pred_probs), p=pred_probs)
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break

    return seed_text

if st.button("Generate Text"):
    if seed:
        result = generate_text(seed)
        st.write("👉", result)
    else:
        st.warning("Please enter seed text")