import time
import concurrent.futures

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import (
    ROLES, DEFAULT_ROLE,
    TOOL_QUERY_MAX_LENGTH, TOOL_QUERY_MIN_LENGTH,
    TOOL_TOPIC_MAX_LENGTH, TOOL_TOPIC_MIN_LENGTH,
    TOOL_TIMEOUTS, DEFAULT_TOOL_TIMEOUT,
    MAX_TOOL_RESULT_LENGTH,
)


# ── Pydantic Schemas ──────────────────────────────────────────────

class DocumentSearchInput(BaseModel):
    """Input schema for the document_search tool."""
    query: str = Field(
        description="The search query to find relevant document chunks.",
        min_length=TOOL_QUERY_MIN_LENGTH,
        max_length=TOOL_QUERY_MAX_LENGTH,
    )

    @field_validator('query')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class DocumentSummarizeInput(BaseModel):
    """Input schema for the document_summarize tool."""
    topic: str = Field(
        description="The topic or focus area for the summary.",
        min_length=TOOL_TOPIC_MIN_LENGTH,
        max_length=TOOL_TOPIC_MAX_LENGTH,
    )

    @field_validator('topic')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class WebSearchInput(BaseModel):
    """Input schema for the web_search tool."""
    query: str = Field(
        description="The search query for the web.",
        min_length=TOOL_QUERY_MIN_LENGTH,
        max_length=TOOL_QUERY_MAX_LENGTH,
    )

    @field_validator('query')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


# ── Timeout Execution ─────────────────────────────────────────────

def execute_with_timeout(func, args, tool_name: str) -> tuple[str, float, bool]:
    """Execute a tool function with timeout.
    Returns (result_str, duration_ms, success).
    """
    timeout = TOOL_TIMEOUTS.get(tool_name, DEFAULT_TOOL_TIMEOUT)
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, args)
        try:
            result = future.result(timeout=timeout)
            duration = (time.time() - start) * 1000
            return str(result), duration, True
        except concurrent.futures.TimeoutError:
            duration = (time.time() - start) * 1000
            return f"Tool '{tool_name}' timed out after {timeout}s.", duration, False
        except Exception as e:
            duration = (time.time() - start) * 1000
            return f"Error executing {tool_name}: {str(e)}", duration, False


# ── Result Truncation ─────────────────────────────────────────────

def _truncate(text: str) -> str:
    if len(text) > MAX_TOOL_RESULT_LENGTH:
        return text[:MAX_TOOL_RESULT_LENGTH] + "\n\n... [Result truncated]"
    return text


# ── Tool Definitions ──────────────────────────────────────────────

def create_document_search_tool(vectorstore):
    @tool(args_schema=DocumentSearchInput)
    def document_search(query: str) -> str:
        """Search uploaded documents for relevant information.

        Use this tool when the user asks a question that can be answered
        from the uploaded documents. Provide a clear search query.

        Args:
            query: The search query to find relevant document chunks.
        """
        docs = vectorstore.similarity_search(query, k=4)
        if not docs:
            return "No relevant documents found for this query."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            header = f"[Chunk {i} | source: {source}"
            if page != "":
                header += f", page: {page}"
            header += "]"
            results.append(f"{header}\n{doc.page_content}")
        return _truncate("\n\n---\n\n".join(results))

    return document_search


def create_document_summarize_tool(vectorstore, model_name: str = "gpt-4o"):
    @tool(args_schema=DocumentSummarizeInput)
    def document_summarize(topic: str) -> str:
        """Summarize the uploaded documents, optionally focused on a topic.

        Use this tool when the user asks for a summary of the documents
        or a broad overview. Provide a topic to focus the summary, or
        use a general term like 'main content' for a full summary.

        Args:
            topic: The topic or focus area for the summary.
        """
        docs = vectorstore.similarity_search(topic, k=20)
        if not docs:
            return "No documents found to summarize."
        combined_text = "\n\n".join(doc.page_content for doc in docs)
        llm = ChatOpenAI(model=model_name, temperature=0)
        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant that summarizes documents. "
                "Provide a clear, structured summary of the following "
                "document content. Focus on: {topic}",
            ),
            ("human", "{text}"),
        ])
        chain = summary_prompt | llm
        response = chain.invoke({"topic": topic, "text": combined_text[:12000]})
        return _truncate(response.content)

    return document_summarize


def create_web_search_tool():
    @tool(args_schema=WebSearchInput)
    def web_search(query: str) -> str:
        """Search the web for information not found in the uploaded documents.

        Use this tool when the user's question requires information that
        is unlikely to be in the uploaded documents, such as current events,
        general knowledge, or external references.

        Args:
            query: The search query for the web.
        """
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return (
                "Web search is not available. "
                "Install duckduckgo-search: pip install duckduckgo-search"
            )
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No web results found for this query."
            formatted = []
            for r in results:
                formatted.append(
                    f"**{r['title']}**\n{r['body']}\nURL: {r['href']}"
                )
            return _truncate("\n\n---\n\n".join(formatted))
        except Exception as e:
            return f"Web search failed: {str(e)}"

    return web_search


# ── Role-Based Tool Factory ───────────────────────────────────────

def get_all_tools(vectorstore, model_name: str = "gpt-4o", role: str = "user") -> list:
    """Create and return tools allowed for the given role."""
    role_config = ROLES.get(role, ROLES[DEFAULT_ROLE])
    allowed = set(role_config["allowed_tools"])

    all_tools = {
        "document_search": create_document_search_tool(vectorstore),
        "document_summarize": create_document_summarize_tool(vectorstore, model_name),
        "web_search": create_web_search_tool(),
    }

    return [t for name, t in all_tools.items() if name in allowed]
