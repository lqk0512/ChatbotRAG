from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader(
    path ="./papers",
    glob = "**/*.pdf",
    loader_cls = PyPDFLoader,
    show_progress = True,
    use_multithreading = True,
)
docs = loader.load()

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",  # Markdown headers
    "```\n",  # Code blocks
    "\n\\*\\*\\*+\n ",  # Markdown horizontal rules
    "\n---+\n ",  # Markdown horizontal rules
    "\n___+\n ",  # Markdown horizontal rules
    "\n\n",  # Paragraph breaks
    "\n",  # Line breaks
    " ",  # Spaces
    "",  # Tabs
]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap = 200,
    add_start_index = True,
    strip_whitespace = True,
    separators = MARKDOWN_SEPARATORS,
)
splits = text_splitter.split_documents(docs)

for doc in splits:
    doc.metadata = {
        "source": doc.metadata.get("source")
    }
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"   # lưu local
)

retriever = vectorstore.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 5, "score_threshold": 0.2}
)

template = (
    "You are a helpful assistant for answering questions based on the following retrieved documents:"
    "RULES:\n"
    "1. Use only the following retrieved documents to answer the question.\n"
    "2. If the retrieved documents do not contain the answer, say you don't know.\n"
    "3. Do not use any information that is not in the retrieved documents.\n"
    "4. If applicable, cites sources as (source:page) using the metadata.\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(template)

llm_base = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
    max_new_tokens=512,
    task="conversational"   
)

llm = ChatHuggingFace(llm=llm_base)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "what is Chemical recycling?"
ans = rag_chain.invoke(question)
print(ans)
