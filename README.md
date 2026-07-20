<!-- ============================= -->
<!-- Header -->
<!-- ============================= -->

<h1 align="center">🧠 Toronto Travel Assistant - RAG Chatbot</h1>

<p align="center">
  <b>Retrieval-Augmented Generation (RAG) Chatbot built with LangChain, OpenAI, Pinecone and Streamlit</b>
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue">
<img src="https://img.shields.io/badge/LangChain-Framework-green">
<img src="https://img.shields.io/badge/OpenAI-GPT--4.1-black">
<img src="https://img.shields.io/badge/Pinecone-VectorDB-orange">
<img src="https://img.shields.io/badge/Streamlit-Frontend-red">

</p>

---

# 📌 Overview

The goal of this project is to develop a domain-specific AI chatbot that combines the reasoning capability of an OpenAI Large Language Model with the efficiency of Pinecone Vector Database using Retrieval-Augmented Generation (RAG).

Instead of relying entirely on the LLM's internal knowledge, the chatbot first retrieves relevant information from custom documents and then generates an accurate, context-aware response.

The chatbot also supports multi-turn conversations using LangChain's conversation memory, enabling users to ask follow-up questions naturally.

---

# ✨ Features

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ OpenAI GPT-4.1 Mini
- ✅ OpenAI Embeddings (text-embedding-3-small)
- ✅ Pinecone Vector Database
- ✅ LangChain
- ✅ Streamlit Chat Interface
- ✅ Conversation Memory
- ✅ Follow-up Question Support
- ✅ Semantic Search
- ✅ Production Style Ingestion Pipeline

---

# 🏗 Overall Architecture

```mermaid
flowchart LR

A[User]

B[Streamlit UI]

C[Conversation Memory]

D[History Aware Retriever]

E[OpenAI Embeddings]

F[Pinecone Vector Database]

G[Top K Relevant Chunks]

H[Prompt Template]

I[OpenAI GPT-4.1]

J[Final Response]

A --> B

B --> C

C --> D

D --> E

E --> F

F --> G

G --> H

H --> I

I --> J
```

---

# 📥 Ingestion Pipeline

```mermaid
flowchart LR

A[Documents TXT PDF]

B[Document Loader]

C[Recursive Text Splitter]

D[OpenAI Embeddings]

E[Pinecone Vector DB]

A --> B

B --> C

C --> D

D --> E
```

---

# 📤 Retrieval Pipeline

```mermaid
flowchart LR

A[User Question]

B[Conversation History]

C[History Aware Retriever]

D[Rewrite Standalone Question]

E[OpenAI Embeddings]

F[Pinecone Similarity Search]

G[Retrieve Top K Chunks]

H[Prompt Template]

I[OpenAI GPT]

J[Answer]

A --> B

B --> C

C --> D

D --> E

E --> F

F --> G

G --> H

H --> I

I --> J
```

---


---

# ⚙️ Technology Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| Backend | LangChain |
| LLM | OpenAI GPT-4.1 Mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Database | Pinecone |
| Programming Language | Python |
| Environment Variables | python-dotenv |

---

# 🚀 Installation

Clone repository

```bash
git clone https://github.com/yourusername/Toronto-RAG-Chatbot.git
```

Go inside project

```bash
cd Toronto-RAG-Chatbot
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Create .env

```env
OPENAI_API_KEY=your_openai_api_key

PINECONE_API_KEY=your_pinecone_api_key
```

---

# ▶️ Run Ingestion Pipeline

```bash
python ingestion.py
```

This script

- Loads documents
- Splits them into chunks
- Generates OpenAI embeddings
- Stores vectors in Pinecone

---

# ▶️ Run Chatbot

```bash
streamlit run streamlitMain.py
```

Open

```
http://localhost:8501
```

---

# 📚 Key Concepts

- Retrieval-Augmented Generation (RAG)
- LangChain
- OpenAI GPT
- Prompt Engineering
- Semantic Search
- Vector Embeddings
- Pinecone
- Conversation Memory
- History Aware Retrieval
- Streamlit

---

---

# 🚀 Future Improvements

To further improve retrieval accuracy and make the chatbot production-ready, the following enhancements can be implemented:

### 🔹 Multi-Query RAG
Generate multiple semantic variations (e.g., 5) of the user's query using an LLM instead of relying on a single query. This improves the chances of retrieving all relevant documents.

### 🔹 Hybrid Search
Combine **Vector Search (Pinecone)** with **Keyword Search (BM25 / Full-Text Search)**. This allows the system to retrieve both semantically similar documents and those containing exact keywords such as product names, IDs, or technical terms.

### 🔹 Reciprocal Rank Fusion (RRF)
Merge results from multiple retrieval strategies (Multi-Query + Hybrid Search) using **Reciprocal Rank Fusion (RRF)** to produce a more robust ranked list of documents.

### 🔹 Semantic Reranking
Use a reranking model (e.g., **Cohere Rerank**, **BGE Reranker**, or **CrossEncoder**) to reorder retrieved documents before passing them to the LLM, ensuring only the most relevant context is used.

---

## 🏗 Proposed Production RAG Pipeline

```mermaid
flowchart LR

A["👤 User Query"]

B["🤖 Generate 5 Query Variations"]

C["🔎 Vector Search (Pinecone)"]

D["🔍 Keyword Search (BM25)"]

E["📊 Reciprocal Rank Fusion (RRF)"]

F["⭐ Semantic Reranker"]

G["📄 Top-k Relevant Chunks"]

H["🧠 OpenAI GPT"]

I["💬 Final Response"]

A --> B

B --> C
B --> D

C --> E
D --> E

E --> F

F --> G

G --> H

H --> I
```

**Benefits**
- ✅ Higher retrieval accuracy
- ✅ Better document recall
- ✅ Improved handling of exact keywords and IDs
- ✅ Reduced hallucinations
- ✅ Better responses for complex and ambiguous queries
- ----

# 🎯 Learning Outcomes

Through this project I learned

- Building production-ready RAG applications
- Working with OpenAI APIs
- Creating semantic search pipelines
- Using Pinecone Vector Database
- Prompt Engineering
- LangChain Retrieval Chains
- Multi-turn Conversation Memory
- Deploying AI applications using Streamlit


<p align="center">

⭐ If you found this repository useful, don't forget to star it!

</p>
