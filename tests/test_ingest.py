"""Unit tests for ingest.py — no real OpenAI or AstraDB calls made."""
import os
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from ingest import chunk_documents, load_vectorstore, create_vectorstore


def test_chunk_documents_splits_large_doc():
    big_text = "word " * 500
    docs = [Document(page_content=big_text, metadata={"source": "test.pdf"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 1200


def test_chunk_documents_small_doc_stays_single():
    docs = [Document(page_content="Short document.", metadata={"source": "test.pdf"})]
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short document."


def test_chunk_documents_preserves_metadata():
    docs = [Document(page_content="Hello world", metadata={"source": "report.pdf", "page": 1})]
    chunks = chunk_documents(docs)
    assert chunks[0].metadata["source"] == "report.pdf"


def test_chunk_documents_empty_input():
    assert chunk_documents([]) == []


@patch("ingest.AstraDBVectorStore")
@patch("ingest.OpenAIEmbeddings")
def test_load_vectorstore_returns_astradb_instance(mock_embeddings, mock_astradb):
    """load_vectorstore() returns a live AstraDB connection using env vars."""
    mock_astradb.return_value = MagicMock()
    with patch.dict(os.environ, {
        "ASTRA_DB_COLLECTION": "test_col",
        "ASTRA_DB_API_ENDPOINT": "https://fake.apps.astra.datastax.com",
        "ASTRA_DB_APPLICATION_TOKEN": "AstraCS:fake",
    }):
        result = load_vectorstore()
    assert result is not None
    mock_astradb.assert_called_once_with(
        collection_name="test_col",
        embedding=mock_embeddings.return_value,
        api_endpoint="https://fake.apps.astra.datastax.com",
        token="AstraCS:fake",
    )


@patch("ingest.AstraDBVectorStore")
@patch("ingest.OpenAIEmbeddings")
def test_create_vectorstore_adds_documents(mock_embeddings, mock_astradb):
    """create_vectorstore() calls add_documents with the provided chunks."""
    mock_vs = MagicMock()
    mock_astradb.return_value = mock_vs
    chunks = [Document(page_content="test chunk", metadata={"source": "test.pdf"})]
    with patch.dict(os.environ, {
        "ASTRA_DB_COLLECTION": "test_col",
        "ASTRA_DB_API_ENDPOINT": "https://fake.apps.astra.datastax.com",
        "ASTRA_DB_APPLICATION_TOKEN": "AstraCS:fake",
    }):
        result = create_vectorstore(chunks)
    mock_vs.add_documents.assert_called_once_with(chunks)
