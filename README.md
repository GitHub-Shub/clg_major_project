# Indian Traffic Law Chatbot

A simple web-based chatbot that answers questions about Indian traffic laws using a Retrieval-Augmented Generation (RAG) pipeline with Ollama’s Gemma-2-9B model, built with Python 3.11 on Windows.

## Overview
This project processes the `MV_Act_English.pdf` file to provide legally accurate responses about traffic laws. It uses a Flask backend, stores law data in a JSON file, and generates embeddings for a FAISS index to power the chatbot, accessible at `http://127.0.0.1:8000/`.


## Features
- Answers queries like "What is the penalty for speeding on a national highway in India?"
- Uses JSON to store traffic law sections instead of a database.
- Employs a RAG pipeline with embeddings for efficient response generation.
- Simple web interface with real-time query processing.

## Prerequisites
- **Python 3.11**
- **Windows OS**
- **Ollama** (for the Gemma-2-9B model)

