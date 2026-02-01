🧠 Next Word Prediction using LSTM

This project implements a Next Word Prediction system using a Long Short-Term Memory (LSTM) neural network trained on Shakespeare’s Hamlet.
The model predicts the most likely next word given a sequence of words and is deployed using a polished Streamlit web application.

📌 Project Overview

Natural Language Processing (NLP) models often need to understand the context of words to predict what comes next.
In this project, an LSTM-based language model learns sequential patterns from literary text and predicts the next word based on prior context.

Key Highlights :

Trained on public-domain Shakespeare text

Uses LSTM for sequence learning

Predicts Top-1 and Top-5 next-word candidates

Deployed with a full-screen Streamlit UI

📊 About Accuracy

The reported accuracy may appear low due to:

Very large vocabulary size

Multiple valid next words in natural language

Word-level prediction being a difficult NLP task

Despite this, qualitative predictions are meaningful, which is expected for language models.
