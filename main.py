import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# LOAD DATA
# -------------------------------
with open("shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Data loaded")

# -------------------------------
# PREPROCESSING
# -------------------------------
text = text.lower()
text = re.sub(r'[^\w\s]', '', text)

# 🔥 IMPORTANT: reduce dataset size (memory fix)
text = text[:50000]

print("Preprocessing done")

# -------------------------------
# TOKENIZATION
# -------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1
print("Total words:", total_words)

# -------------------------------
# CREATE SEQUENCES
# -------------------------------
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

print("Sequences created")

# -------------------------------
# PADDING
# -------------------------------
max_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# -------------------------------
# SPLIT X AND y
# -------------------------------
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("Data ready for training")

# -------------------------------
# MODEL
# -------------------------------
model = Sequential([
    Embedding(total_words, 100, input_length=max_len-1),
    LSTM(150, return_sequences=True),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

# 🔥 IMPORTANT: changed loss function
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Model built")

# -------------------------------
# TRAIN
# -------------------------------
early_stop = EarlyStopping(monitor='loss', patience=3)

model.fit(X, y, epochs=10, callbacks=[early_stop])

print("Training complete")

# -------------------------------
# TEXT GENERATION
# -------------------------------
def generate_text(seed_text, next_words=20):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break

    return seed_text


print("\nGenerated Text:")
print(generate_text("to be or not to", 15))
print("\n--- OUTPUTS ---")

print("\nSeed: to be or not to")
print(generate_text("to be or not to", 15))

print("\nSeed: love is")
print(generate_text("love is", 15))

print("\nSeed: king of")
print(generate_text("king of", 15))