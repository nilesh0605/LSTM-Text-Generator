LSTM TEXT GENERATION PROJECT

---

🔹 PROJECT DESCRIPTION
This project implements a text generation model using LSTM (Long Short-Term Memory). The model is trained on Shakespeare’s dataset and is capable of generating new text based on a given seed input. The goal is to predict the next word in a sequence and generate coherent text.

---

🔹 DATASET
Dataset Used: Shakespeare’s Complete Works
Source: Project Gutenberg
Link: https://www.gutenberg.org/files/100/100-0.txt

---

🔹 METHODOLOGY

1. Data Preprocessing:

* Converted all text to lowercase
* Removed punctuation and special characters
* Tokenized text into sequences of words
* Created input-output pairs for training

2. Model Design:

* Used Embedding Layer to convert words into vectors
* Used two LSTM layers to learn sequence patterns
* Used Dense layer with softmax activation to predict next word

3. Model Training:

* Optimizer: Adam
* Loss Function: sparse categorical crossentropy
* Applied EarlyStopping to prevent overfitting

4. Text Generation:

* Provided a seed input
* Model predicts next word step-by-step
* Generated readable text output

---

🔹 SAMPLE OUTPUT

Seed: to be or not to
Output: to be or not to my love doth thee thee...

Seed: love is
Output: love is the heart that speaks within...

Seed: king of
Output: king of england shall be the lord...

---

🔹 IMPROVEMENTS

* Reduced memory usage by using sparse categorical crossentropy instead of one-hot encoding
* Improved text generation by adjusting epochs and dataset size
* Added probabilistic sampling to reduce repetition

---

🔹 HOW TO RUN

1. Install dependencies:
   pip install tensorflow numpy

2. Run the program:
   python main.py

---

🔹 AUTHOR
Name: Nilesh
Project: Generative AI - LSTM Text Generator

---
