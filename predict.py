import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("next_word_lstm.h5")

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

index_to_word = {index: word for word, index in tokenizer.word_index.items()}

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predictions = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predictions, axis=1)[0]

    return index_to_word.get(predicted_index, None)


