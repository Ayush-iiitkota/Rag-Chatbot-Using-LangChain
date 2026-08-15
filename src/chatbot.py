import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
load_dotenv(override=True)
INDEX_NAME = "document-rag"

# ==========================================================
# Embeddings
# ==========================================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ==========================================================
# LLM
# ==========================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# ==========================================================
# Prompts
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

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful assistant.
Answer ONLY using the retrieved context from the user's uploaded documents.

If the answer is unavailable in the context, reply exactly:
"I don't know based on the provided documents."

Context:
{context}
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

from pinecone import Pinecone, ServerlessSpec

# ==========================================================
# Ingestion Function
# ==========================================================
def ingest_pdf_to_pinecone(file_path: str, namespace: str):
    """Loads a PDF, splits it, and uploads to Pinecone with a specific namespace."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Ensure Pinecone index exists
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        import time
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
    
    # Upload to Pinecone
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )
    return len(chunks)

# ==========================================================
# Conversation Memory
# ==========================================================
store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ==========================================================
# ChatBot Wrapper
# ==========================================================
class ChatBot:
    def __init__(self, namespace: str):
        # Initialize VectorStore with namespace
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        history_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_prompt,
        )
        
        document_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
        )
        
        rag_chain = create_retrieval_chain(
            history_retriever,
            document_chain,
        )
        
        self.chatbot = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )