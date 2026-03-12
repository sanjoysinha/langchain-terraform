from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


def build_rag_chain(vectorstore, model_name: str = "gpt-4o"):
    """Build a conversational RAG chain with chat history support."""
    llm = ChatOpenAI(model=model_name, temperature=0)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # Step 1: History-aware retriever — reformulates the question given chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might "
            "reference context in the chat history, formulate a standalone "
            "question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed "
            "and otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Step 2: Question-answer chain
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, say that you don't "
            "know. Keep the answer concise."
            "\n\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Step 3: Full retrieval chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def format_chat_history(messages: list[dict]) -> list:
    """Convert Streamlit message dicts to LangChain message objects."""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history
