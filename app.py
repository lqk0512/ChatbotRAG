import streamlit as st
import os
from rag import ask
from ingest import ingest

st.set_page_config(layout="wide")

# SIDEBAR
st.sidebar.title("📂 Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        path = f"data/{file.name}"
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        ingest(path)

    st.sidebar.success("Documents added!")

# CHAT UI
st.title("📚 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    answer = ask(question)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })