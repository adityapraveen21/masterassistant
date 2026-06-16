import streamlit as st
from masterassistant import app

st.set_page_config(
    page_title="Master Assistant",
    page_icon="🤖"
)

st.title("🤖 Master Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Run LangGraph app
    result = app.invoke(
        {
            "user_input": user_input,
            "route": "",
            "response": "",
            "chat_history": st.session_state.chat_history
        }
    )

    assistant_response = result["response"]

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response
        }
    )

    # Update conversation history
    st.session_state.chat_history += (
        f"\nUser: {user_input}\n"
        f"Assistant: {assistant_response}\n"
    )