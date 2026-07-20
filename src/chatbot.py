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
        self.chatbot = chatbot
