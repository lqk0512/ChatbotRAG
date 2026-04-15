from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True
)

def ingest(file_path: str):
    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Split
    splits = text_splitter.split_documents(docs)

    # 3. Giữ metadata (QUAN TRỌNG cho citation)
    for doc in splits:
        doc.metadata = {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "start_index": doc.metadata.get("start_index")
        }

    # 4. Load DB (append mode)
    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    print(f"✅ Added {file_path}")