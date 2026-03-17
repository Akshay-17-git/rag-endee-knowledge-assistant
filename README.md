# 🧠 AI Knowledge Assistant — Endee Vector Database + RAG

> A production-ready semantic search and Retrieval-Augmented Generation (RAG) system powered by the **Endee** vector database.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Endee](https://img.shields.io/badge/Endee-Vector%20DB-6366f1.svg)](https://github.com/endee-io/endee)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

This project is an **AI-powered knowledge assistant** that allows users to upload documents (PDFs, text files) and interact with them using natural language queries. It uses **semantic search** and **Retrieval-Augmented Generation (RAG)** to provide context-aware answers grounded in the uploaded content.

### Problem Statement

Traditional search relies on keyword matching — failing to capture semantic meaning. When working with unstructured documents (research papers, notes, resumes), keyword search returns irrelevant or incomplete results.

### Solution

This system converts documents into **vector embeddings**, stores them in **Endee**, and retrieves semantically relevant chunks for any query. A language model (Mistral) then synthesizes these chunks into a coherent, accurate answer.

---

## 🏗️ System Architecture

```
User Query
   │
   ▼
SentenceTransformers (all-MiniLM-L6-v2)   ◄── Embedding Model
   │
   ▼
Endee Vector Database ──────────────────────── Core: cosine similarity search
   │
   ▼
Top-K Relevant Chunks (retrieved context)
   │
   ▼
Mistral API (mistral-small-latest) ─────────── Answer Generation
   │
   ▼
Natural Language Answer → User
```

**Data Ingestion Flow:**
```
Document (PDF/TXT)
   │
   ▼
Text Extraction → Sentence Splitting → Overlapping Chunks
   │
   ▼
SentenceTransformer Embeddings (384-dim vectors)
   │
   ▼
Endee.insert(id, vector, metadata)
```

---

## 🧠 How Endee is Used

**Endee** serves as the backbone vector database for this project:

| Operation | Description |
|-----------|-------------|
| `EndeeClient.insert(id, vector, metadata)` | Store embedded document chunks |
| `EndeeClient.insert_batch(records)` | Bulk insertion for efficiency |
| `EndeeClient.search(query_vector, top_k)` | Cosine similarity retrieval |
| `EndeeClient.delete(id)` | Remove specific vectors |
| `EndeeClient.clear()` | Reset the collection |
| `EndeeClient.info()` | Collection statistics |

**Why Endee?**
- High-performance similarity search for production AI systems
- Efficient handling of high-dimensional embedding vectors (384-dim)
- Clean API surface that integrates naturally into RAG pipelines
- Scalable to large document collections

The `EndeeClient` class in `utils/endee_client.py` mirrors the production Endee SDK interface, making it trivial to swap in the real SDK once the project is deployed.

---

## 🌟 Key Features

- **🔍 Semantic Search** — Find relevant content based on meaning, not keywords
- **🤖 RAG Pipeline** — Retrieval-Augmented Generation using Mistral
- **📄 Document Q&A** — Ask anything about uploaded documents
- **🧾 Context-aware Summarization** — Summarize specific topics within documents
- **🎯 Resume Interview Prep** — Generate targeted interview questions from a resume
- **⚡ Endee Vector Store** — Fast similarity search with cosine distance
- **📦 Batch Indexing** — Efficient multi-document ingestion

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| UI | Streamlit |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Database | **Endee** |
| LLM | Mistral API (`mistral-small-latest`) |
| PDF Parsing | pdfplumber |
| Data | NumPy, Pandas |

---

## ▶️ Setup and Execution

### Prerequisites
- Python 3.9+
- Mistral API key (free tier available at [console.mistral.ai](https://console.mistral.ai))

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/ai-knowledge-assistant-endee.git
cd ai-knowledge-assistant-endee
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set API Key (optional — app prompts for it in the UI)
```bash
export MISTRAL_API_KEY=your_mistral_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
ai-knowledge-assistant-endee/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── utils/
    ├── __init__.py
    ├── endee_client.py         # Endee vector DB client & operations
    ├── embeddings.py           # SentenceTransformer embedding generation
    ├── document_processor.py  # Text chunking and preprocessing
    └── llm.py                  # Mistral API integration for RAG
```

---

## 🔄 RAG Workflow

### Step 1: Document Ingestion
1. Upload PDF or text file
2. Extract and clean text
3. Split into overlapping chunks (~150 words each)
4. Generate 384-dim embeddings via SentenceTransformers
5. Store `(id, vector, metadata)` in **Endee**

### Step 2: Query Processing
1. User submits a natural language query
2. Query is converted to an embedding vector
3. **Endee** performs cosine similarity search
4. Top-K most relevant chunks are retrieved

### Step 3: Answer Generation
1. Retrieved chunks form the context window
2. Context + query sent to Mistral API
3. Mistral generates a grounded, contextual answer
4. Answer displayed with source citations

---

## 📊 Evaluation Criteria

| Metric | Approach |
|--------|----------|
| Retrieval Accuracy | Cosine similarity scores of retrieved chunks |
| Response Quality | Relevance and completeness of Mistral answers |
| Latency | Sub-second retrieval for <10K chunks |
| Scalability | Batch ingestion, configurable chunk size |

---

## 🔮 Future Improvements

- [ ] Hybrid search: BM25 + vector similarity (reciprocal rank fusion)
- [ ] Multi-collection support for document namespacing
- [ ] Streaming responses from Mistral
- [ ] Evaluation dashboard with RAGAS metrics
- [ ] Export chat history as PDF
- [ ] Authentication and multi-user support

---

## 📎 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ using <strong>Endee Vector DB</strong> + Mistral + Streamlit</p>
