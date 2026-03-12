import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore


def load_documents(file_paths: list[str]) -> list:
    """Load PDF and TXT files and return a flat list of Document objects."""
    documents = []
    for path in file_paths:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.lower().endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        documents.extend(loader.load())
    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks: list) -> AstraDBVectorStore:
    """Create and populate an AstraDB vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = AstraDBVectorStore(
        collection_name=os.environ["ASTRA_DB_COLLECTION"],
        embedding=embeddings,
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    )
    vectorstore.add_documents(chunks)
    return vectorstore


def load_vectorstore() -> AstraDBVectorStore:
    """Connect to the AstraDB vector store. Always returns a live connection."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return AstraDBVectorStore(
        collection_name=os.environ["ASTRA_DB_COLLECTION"],
        embedding=embeddings,
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    )


def ingest_documents(file_paths: list[str]) -> AstraDBVectorStore:
    """Full pipeline: load documents, chunk, embed, and upsert to AstraDB."""
    documents = load_documents(file_paths)
    chunks = chunk_documents(documents)
    return create_vectorstore(chunks)
