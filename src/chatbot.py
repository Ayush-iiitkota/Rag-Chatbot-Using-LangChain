import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import create_stuff_documents_chain


# ==========================================================
# Environment
# ==========================================================

load_dotenv()

INDEX_NAME = "toronto-rag"

# ==========================================================
# Embeddings
# ==========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ==========================================================
# Vector Store
# ==========================================================

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ==========================================================
# LLM
# ==========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# ==========================================================
# Prompt to rewrite follow-up questions
# ==========================================================

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question into a standalone question. "
            "Do not answer it. If already standalone, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt,
)

# ==========================================================
# QA Prompt
# ==========================================================

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Toronto Travel Assistant.

Answer ONLY using the retrieved context.

If the answer is unavailable, reply:

"I don't know based on the provided documents."

Context:
{context}
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

document_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

rag_chain = create_retrieval_chain(
    history_retriever,
    document_chain,
)

# ==========================================================
# Conversation Memory
# ==========================================================

store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chatbot = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


    # ==========================================================
# ChatBot Wrapper for Streamlit
# ==========================================================

class ChatBot:
    def __init__(self):
        self.chatbot = chatbot"""
=========================================================
CHATBOT / RETRIEVAL PIPELINE
=========================================================

This file performs ONLY retrieval.

Workflow:
User Question
      |
      V
History Aware Retriever
      |
      V
Retrieve Relevant Chunks from Pinecone
      |
      V
Prompt + Context
      |
      V
GPT-4.1 Mini
      |
      V
Answer

Conversation history is maintained so the chatbot
can answer follow-up questions.

=========================================================
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_NAME = "toronto-rag"

# ==========================================================
# Initialize OpenAI Embedding Model
# ==========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)

# ==========================================================
# Connect to Pinecone Vector Store
# ==========================================================

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)

# ==========================================================
# Initialize LLM
# ==========================================================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

# ==========================================================
# Prompt 1
#
# Rewrite follow-up questions into standalone questions.
#
# Example
#
# User:
# Tell me about CN Tower.
#
# User:
# When does it open?
#
# becomes
#
# When does the CN Tower open?
#
# ==========================================================

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are given a chat history and the latest user question.

Rewrite the latest question into a standalone question.

Do NOT answer the question.

Only rewrite it if necessary.

If the question is already standalone,
return it unchanged.
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt,
)

# ==========================================================
# Prompt 2
#
# Final Question Answering Prompt
# ==========================================================

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Toronto Travel Assistant.

Answer ONLY using the retrieved context.

If the answer cannot be found in the context,
reply exactly:

"I don't know based on the provided documents."

Keep answers concise.

Context:

{context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# ==========================================================
# Combine Retrieved Documents with Prompt
# ==========================================================

question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

# ==========================================================
# Create Retrieval Chain
# ==========================================================

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)

# ==========================================================
# Conversation Memory
#
# Stores chat history for each session.
#
# Replace with Redis/Database in production.
# ==========================================================

chat_store = {}


def get_session_history(session_id: str):

    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()

    return chat_store[session_id]


# ==========================================================
# Wrap Chain with Conversation Memory
# ==========================================================

chatbot = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ==========================================================
# Chat Loop
# ==========================================================

print("=" * 60)
print("Toronto Travel Assistant")
print("Type 'exit' to quit.")
print("=" * 60)

SESSION_ID = "user_001"

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    response = chatbot.invoke(
        {"input": question},
        config={
            "configurable": {
                "session_id": SESSION_ID
            }
        },
    )

    print("\nBot :", response["answer"])
