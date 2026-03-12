"""Unit tests for chain.py — no real OpenAI calls made."""
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from chain import format_chat_history


def test_format_chat_history_empty():
    assert format_chat_history([]) == []


def test_format_chat_history_user_message():
    messages = [{"role": "user", "content": "What is RAG?"}]
    result = format_chat_history(messages)
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "What is RAG?"


def test_format_chat_history_assistant_message():
    messages = [{"role": "assistant", "content": "RAG is retrieval augmented generation."}]
    result = format_chat_history(messages)
    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "RAG is retrieval augmented generation."


def test_format_chat_history_mixed():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "What is LangChain?"},
    ]
    result = format_chat_history(messages)
    assert len(result) == 3
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)
    assert isinstance(result[2], HumanMessage)


def test_format_chat_history_ignores_unknown_roles():
    messages = [
        {"role": "system", "content": "You are a bot"},
        {"role": "user", "content": "Hi"},
    ]
    result = format_chat_history(messages)
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)


@patch("chain.ChatOpenAI")
@patch("chain.create_retrieval_chain")
@patch("chain.create_history_aware_retriever")
@patch("chain.create_stuff_documents_chain")
def test_build_rag_chain_returns_chain(
    mock_stuff, mock_history, mock_retrieval, mock_llm
):
    from chain import build_rag_chain

    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()
    mock_retrieval.return_value = MagicMock()

    chain = build_rag_chain(mock_vectorstore)

    assert chain is not None
    mock_vectorstore.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={"k": 4}
    )
