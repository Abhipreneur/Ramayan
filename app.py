# # 📦 Step 1: Install Dependencies
# ! pip install keras tensorflow
# ! pip install streamlit

# 🧹 Step 2: Import Libraries
import streamlit as st

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# 📄 Step 3: Sample Corpus (replace this with your own dataset!)
corpus = [
    "I am going to the store",
    "I am going to the park",
    "He is reading a book",
    "She is playing in the garden",
    "They are going for a walk"
]

# 🧼 Step 4: Tokenization and Sequence Creation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split input and label
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)


# 🧠 Step 5: Define the LSTM Model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_len - 1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# 🏋️ Step 6: Train the Model
model.fit(X, y, epochs=200, verbose=1)

# 🔮 Step 7: Prediction Function
def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""



# Streamlit UI
st.title("Next Word Predictor")
user_input = st.text_input("Enter your phrase:", "I am going")

if st.button("Predict"):
    next_word = predict_next_word(user_input)
    st.success(f"Next word prediction: **{next_word}**")





