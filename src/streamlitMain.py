import streamlit as st
from RAG_ChatBot import ChatBot

# Load chatbot only once
@st.cache_resource
def load_chatbot():
    return ChatBot()

bot = load_chatbot()

st.set_page_config(
    page_title="Toronto Travel Assistant",
    page_icon="✈️"
)

st.title("✈️ Toronto Travel Assistant")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":"assistant",
            "content":"Hello! I'm your Toronto Travel Assistant. Ask me anything."
        }
    ]

# Display history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            response = bot.chatbot.invoke(

                {"input":prompt},

                config={
                    "configurable":{
                        "session_id":"streamlit_user"
                    }
                }

            )

            answer = response["answer"]

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )