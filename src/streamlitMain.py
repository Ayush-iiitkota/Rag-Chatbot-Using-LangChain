import streamlit as st
import os
import uuid
from chatbot import ChatBot, ingest_pdf_to_pinecone

st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="centered"
)

# Initialize Session State
if "namespace" not in st.session_state:
    st.session_state.namespace = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    st.session_state.bot = None

st.title("📄 PDF Assistant")
st.caption("Upload your PDFs and ask questions exclusively from them.")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                os.makedirs("uploads", exist_ok=True)
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Ingest
                    chunks = ingest_pdf_to_pinecone(file_path, st.session_state.namespace)
                    total_chunks += chunks
                    
                    # Clean up
                    os.remove(file_path)
                
                # Initialize ChatBot with the new namespace
                st.session_state.bot = ChatBot(namespace=st.session_state.namespace)
                st.success(f"Processed {len(uploaded_files)} files into {total_chunks} chunks.")
        else:
            st.warning("Please upload at least one PDF.")

# Display Previous Messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            if "context" in message and message["context"]:
                with st.expander("View Retrieved Context"):
                    for doc in message["context"]:
                        st.markdown(f"**Source: {doc.metadata.get('source', 'Unknown')}**")
                        st.text(doc.page_content)

# User Input
if not st.session_state.bot:
    st.info("Please upload and process PDFs in the sidebar to start chatting.")
else:
    prompt = st.chat_input("Ask a question about the PDFs...")
    if prompt:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = st.session_state.bot.chatbot.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": st.session_state.namespace}}
                )
                answer = response["answer"]
                context_docs = response.get("context", [])

                st.markdown(answer)
                if context_docs:
                    with st.expander("View Retrieved Context"):
                        for doc in context_docs:
                            st.markdown(f"**Source: {doc.metadata.get('source', 'Unknown')}**")
                            st.text(doc.page_content)

        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "context": context_docs
        })