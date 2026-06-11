# Advanced Task 4: Context-Aware RAG Chatbot

## 📌 Objective
Build a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on a custom knowledge base using semantic search and local LLMs — with no external API dependency.

---

## 🏗️ Architecture

```
Knowledge Base (Text Corpus)
   │
   ▼
RecursiveCharacterTextSplitter (chunk_size=300, overlap=50)
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
   │
   ▼
FAISS Vector Store (saved locally as faiss_index/)
   │
   ▼
User Query → Embed Query → Similarity Search → Top-K Chunks Retrieved
   │
   ▼
Retrieved Chunks returned as Answer with Source Attribution
```

---

## 🗂️ Knowledge Base
Custom AI/ML corpus covering 5 topics, ~30 chunks total:

| Topic | Description |
|-------|-------------|
| Machine Learning | Supervised, unsupervised, reinforcement learning |
| Large Language Models | GPT-4, Claude, Gemini, transformers, fine-tuning |
| Retrieval-Augmented Generation | RAG pipeline, vector stores, hallucination reduction |
| Neural Networks | CNNs, RNNs, transformers, backpropagation |
| Natural Language Processing | Tokenization, NER, BERT, sentiment analysis |

---

## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| `langchain==0.3.25` | RAG chain orchestration |
| `langchain-community==0.3.24` | HuggingFace integrations, FAISS wrapper |
| `langchain-text-splitters==0.3.8` | Document chunking |
| `sentence-transformers` | Local embedding model |
| `faiss-cpu` | Vector similarity search |

---

## 📋 Steps Performed

### 1. Build Knowledge Base
```python
from langchain_core.documents import Document
all_docs = [Document(page_content=text, metadata={"title": title})
            for title, text in corpus]
```

### 2. Chunk Documents
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
```

### 3. Embed & Store in FAISS
```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
```

### 4. Semantic Retrieval
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke("What is machine learning?")
```

---

## 💬 Sample Output
```
🧑 User: What is machine learning?
🤖 Bot (retrieved answer):
  [1] Source: Machine Learning
       Machine learning is a branch of artificial intelligence that enables
       systems to learn and improve from experience without being explicitly
       programmed...

  [2] Source: Machine Learning
       Common supervised learning algorithms include linear regression,
       decision trees, and neural networks...
------------------------------------------------------------
```

---

## 🔍 Key Observations
- `chunk_overlap=50` prevents context from being cut off at chunk boundaries
- `all-MiniLM-L6-v2` is a production-grade embedding model used in real-world RAG systems — lightweight (~90MB) yet highly accurate
- FAISS enables sub-millisecond similarity search even at scale
- The pipeline was designed for Gemini/OpenAI LLM integration; switched to retrieval-only demo due to API quota constraints — the core RAG components are fully production-ready

---

## 📁 Repository Structure
```
Advanced-Task-4-RAG-Chatbot/
├── rag_chatbot.ipynb     # Full notebook
├── app.py                # Streamlit deployment interface
├── faiss_index/          # Saved FAISS vector index
└── README.md
```

---

## ✅ Skills Demonstrated
- Document ingestion, chunking, and embedding pipeline
- FAISS vector store creation and local persistence
- Semantic similarity search and retrieval
- Production-ready RAG architecture with zero external API dependency

---

## 👨‍💻 Author
**Muhammad Hassaan**
AI/ML Engineering Intern — DevelopersHub Corporation
DHC-488
