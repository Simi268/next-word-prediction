import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Next Word Prediction | LSTM",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    input {
        font-size: 18px !important;
    }

    button {
        font-size: 18px !important;
        padding: 10px 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(base_dir, "next_word_lstm.h5"))
    with open(os.path.join(base_dir, "tokenizer.pickle"), "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

index_to_word = {index: word for word, index in tokenizer.word_index.items()}
max_sequence_len = model.input_shape[1] + 1

def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]
    token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predictions = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predictions, axis=1)[0]

    return index_to_word.get(predicted_index, "")

def predict_top_k(text, k=5):
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]
    token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predictions = model.predict(token_list, verbose=0)[0]
    top_k_indices = predictions.argsort()[-k:][::-1]

    return [index_to_word.get(i, "") for i in top_k_indices]

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🧠 Next Word Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "LSTM Language Model trained on Shakespeare (Hamlet)"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

left, center, right = st.columns([1, 6, 1])
with center:
    st.subheader("✍️ Enter Text")
    input_text = st.text_input(
        "",
        value="to be or not to",
        label_visibility="collapsed"
    )


col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    predict_clicked = st.button("🔮 Predict Next Word", use_container_width=True)

st.markdown("---")

if predict_clicked:
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(input_text)
        top_words = predict_top_k(input_text, k=5)

        left, right = st.columns([1, 2])

        with left:
            st.markdown("### ✅ Predicted Word")
            st.markdown(
                f"""
                <div style="
                    background-color:#0E1117;
                    padding:30px;
                    border-radius:12px;
                    text-align:center;
                    font-size:28px;
                    font-weight:bold;
                    border:1px solid #262730;
                ">
                    {next_word}
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            st.markdown("### 📊 Top-5 Predictions")
            for i, word in enumerate(top_words, start=1):
                st.markdown(
                    f"""
                    <div style="
                        padding:12px;
                        margin-bottom:8px;
                        border-radius:8px;
                        border:1px solid #262730;
                        font-size:16px;
                    ">
                        <strong>{i}.</strong> {word}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Built with TensorFlow • LSTM • Streamlit"
    "</p>",
    unsafe_allow_html=True
)

