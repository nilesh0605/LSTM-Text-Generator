# 🔥 LSTM Text Generator (Generative AI Project)

## 📌 Project Overview

This project implements a text generation system using an LSTM (Long Short-Term Memory) neural network. The model is trained on Shakespeare’s Complete Works dataset and is capable of generating new text based on a given seed input.

The project covers the complete pipeline of Generative AI including preprocessing, model building, training, and deployment with a user interface.

---

## 🚀 Features

* Text preprocessing (lowercase conversion & punctuation removal)
* Tokenization and sequence generation
* LSTM-based deep learning model
* Memory-efficient training using sparse categorical crossentropy
* Text generation using seed input
* Streamlit-based interactive UI
* Temperature control for randomness tuning
* Top-K sampling for better text quality

---

## 📂 Dataset

* **Dataset Used:** Shakespeare’s Complete Works
* **Source:** Project Gutenberg
* **Link:** https://www.gutenberg.org/files/100/100-0.txt

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Streamlit

---

## 🧠 Model Architecture

* Embedding Layer
* LSTM Layer (150 units)
* LSTM Layer (100 units)
* Dense Layer with Softmax activation

---

## 🏋️ Training Details

* Loss Function: Sparse Categorical Crossentropy
* Optimizer: Adam
* Early Stopping used to prevent overfitting
* Dataset size optimized for memory efficiency

---

## 🎯 Text Generation

The model generates text by predicting the next word in a sequence iteratively.

### Techniques Used:

* Temperature Sampling (controls randomness)
* Top-K Sampling (selects best probable words)

---

## ✨ Sample Outputs

**Seed Input:**
`to be or not to`

**Generated Output:**
`to be or not to you thou leisure doth thy love are rare...`

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install tensorflow numpy streamlit
```

### 2. Train the model

```bash
python main.py
```

### 3. Run UI

```bash
streamlit run app.py
```

---

## 📦 Project Structure

```
LSTM-Text-Generator/
│── main.py
│── app.py
│── shakespeare.txt
│── lstm_model.h5
│── tokenizer.pkl
│── README.md
```

---

## 💡 Improvements & Learnings

* Reduced memory usage by avoiding one-hot encoding
* Improved output quality using sampling techniques
* Learned sequence modeling using LSTM networks
* Built an end-to-end AI project from scratch

---

## 👨‍💻 Author

**Nilesh**
Aspiring AI/ML Developer

---

## ⭐ Conclusion

This project demonstrates how LSTM networks can be used for sequence prediction and generative AI tasks. While the generated text may not be perfectly grammatical, it successfully captures patterns from the training data.

---

⭐ If you like this project, consider giving it a star!
