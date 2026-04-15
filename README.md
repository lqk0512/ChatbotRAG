# RAG Chatbot for Document Question Answering

This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions based strictly on the document content. The system retrieves relevant context from the documents and uses a language model to generate accurate, source-grounded answers.

Live Demo:
https://chatbotrag-4x87sijsqpuovg32mube8m.streamlit.app/

---

## Overview

The application is designed to replicate the core functionality of systems like NotebookLM, focusing on document-grounded reasoning. It ensures that all responses are derived only from retrieved content, reducing hallucination and improving reliability.

---

## Features

* Upload and process multiple PDF documents
* Automatic text extraction and chunking
* Semantic search using vector embeddings
* Retrieval-Augmented Generation pipeline
* Answers constrained to retrieved context only
* Source citation with document and page reference
* Ability to reset and manage uploaded documents
* Lightweight web interface using Streamlit

---

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB (vector database)
* HuggingFace Inference API
* Embedding model: BAAI/bge-small-en-v1.5
* Language model: Qwen2.5-7B-Instruct

---

## System Architecture

The system follows a standard RAG pipeline:

1. Document Ingestion

   * Load PDF files
   * Extract text
   * Split into overlapping chunks

2. Embedding

   * Convert text chunks into vector representations

3. Storage

   * Store embeddings in ChromaDB

4. Retrieval

   * Retrieve top-k relevant chunks based on user query

5. Generation

   * Pass retrieved context to LLM
   * Generate answer constrained by context

---

## Project Structure

```
.
├── app.py              # Streamlit user interface
├── rag.py              # RAG pipeline and query handling
├── ingest.py           # Document ingestion and indexing
├── requirements.txt
├── data/               # Uploaded documents (runtime)
└── chroma_db/          # Vector database (ignored in git)
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/lqk0512/ChatbotRAG.git
cd ChatbotRAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

### 4. Run the application

```bash
streamlit run app.py
```

---

## Deployment

The application is deployed using Streamlit Cloud.

Steps to deploy:

1. Push the project to GitHub
2. Connect the repository on Streamlit Cloud
3. Add the following secret in the dashboard:

```
HUGGINGFACEHUB_API_TOKEN=your_token
```

4. Deploy the application

---

## Limitations

* Performance depends on PDF text quality
* HuggingFace free tier may introduce latency
* Data is not persistent across sessions on Streamlit Cloud
