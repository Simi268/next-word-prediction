import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


input_sequences = np.load("input_sequences.npy")


with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

total_words = tokenizer.num_words
max_sequence_len = input_sequences.shape[1]


X = input_sequences[:, :-1]
y = input_sequences[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len - 1),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

model.save("next_word_lstm.h5")
print(" Model saved as next_word_lstm.h5")

