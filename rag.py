from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

try:
    import streamlit as st
    token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 🔥 Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# 🔥 Load vector DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.2}
)

# 🔥 Format context + citation
def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source')}:{d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )

# 🔥 Prompt (NotebookLM style)
template = """
You are a helpful assistant.

Rules:
- Use ONLY the provided context
- If not found → say "I don't know"
- ALWAYS cite sources like (source:page)

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm_base = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    max_new_tokens=256,
    task="conversational",
    huggingfacehub_api_token=token,
)

llm = ChatHuggingFace(llm=llm_base)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 🔥 API
def ask(question: str):
    return rag_chain.invoke(question)