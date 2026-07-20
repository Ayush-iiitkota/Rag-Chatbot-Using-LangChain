"""
=========================================================
INGESTION PIPELINE
=========================================================

Purpose:
--------
This script loads documents, splits them into chunks,
creates embeddings using OpenAI, and stores them in
Pinecone.

Run this script ONLY when:
1. New documents are added
2. Existing documents are modified

You DO NOT run this script for every user query.

=========================================================
"""

import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "toronto-rag"

# =========================================================
# Step 1 : Load Documents
# =========================================================

print("Loading documents...")

loader = TextLoader(
    "./materials/torontoTravelAssistant.txt",
    encoding="utf-8"
)

documents = loader.load()

print(f"Loaded {len(documents)} document(s).")

# =========================================================
# Step 2 : Split Documents into Chunks
# =========================================================

print("Splitting documents...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# =========================================================
# Step 3 : Initialize OpenAI Embedding Model
# =========================================================

print("Initializing embedding model...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# OpenAI text-embedding-3-small produces 1536-dimensional vectors.

# =========================================================
# Step 4 : Connect to Pinecone
# =========================================================

print("Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

# =========================================================
# Step 5 : Create Index (Only First Time)
# =========================================================

existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:

    print("Creating Pinecone index...")

    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

else:
    print("Index already exists.")

# =========================================================
# Step 6 : Store Chunks into Pinecone
# =========================================================

print("Uploading vectors to Pinecone...")

vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

# =========================================================
# Finished
# =========================================================

print("\n========================================")
print("Ingestion Completed Successfully!")
print(f"Index Name : {INDEX_NAME}")
print(f"Chunks Uploaded : {len(chunks)}")
print("========================================")