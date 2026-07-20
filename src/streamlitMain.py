import streamlit as st
from chatbot import ChatBot

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Toronto Travel Assistant",
    page_icon="✈️",
    layout="centered"
)

# ==========================================================
# Load Chatbot (Runs Only Once)
# ==========================================================

@st.cache_resource
def load_chatbot():
    return ChatBot()

bot = load_chatbot()

# ==========================================================
# Title
# ==========================================================

st.title("✈️ Toronto Travel Assistant")
st.caption("Powered by Groq • HuggingFace • Pinecone • LangChain")

# ==========================================================
# Initialize Chat History
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Toronto Travel Assistant.\n\n"
                "Ask me anything about Toronto based on the provided documents."
            )
        }
    ]

# ==========================================================
# Display Previous Messages
# ==========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================================
# User Input
# ==========================================================

prompt = st.chat_input("Ask a question...")

if prompt:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            response = bot.chatbot.invoke(
                {"input": prompt},
                config={
                    "configurable": {
                        "session_id": "streamlit_user"
                    }
                }
            )

            answer = response["answer"]

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
