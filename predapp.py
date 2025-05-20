#  Disable TensorFlow
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
from transformers import pipeline, set_seed

# Reproducibility
set_seed(42)

# Load GPT-2 generation pipeline (PyTorch only)
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2", framework="pt")  # Explicitly use PyTorch

generator = load_generator()

# UI
st.title("📜 Ramayan Sentence Generator")
st.markdown("Generate Ramayan-style text using GPT-2. Start with a phrase and let the model continue.")

user_input = st.text_input("📝 Enter starting text:", "Ram went to the forest")
max_len = st.slider("📏 Max length", 50, 200, 100)
num_return_sequences = st.slider("🔁 Variations", 1, 5, 1)

if st.button("✨ Generate"):
    with st.spinner("Generating..."):
        results = generator(user_input, max_length=max_len, num_return_sequences=num_return_sequences)
        for i, result in enumerate(results):
            st.subheader(f"📖 Output {i+1}")
            st.write(result["generated_text"])
