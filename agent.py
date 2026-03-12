from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from tools import get_all_tools, execute_with_timeout
from guardrails import HARDENED_SYSTEM_PROMPT


def build_agent(vectorstore, model_name: str = "gpt-4o", role: str = "user"):
    """Build the tool-calling agent components.

    Returns (llm_with_tools, tools_list, tool_map).
    """
    tools = get_all_tools(vectorstore, model_name, role=role)
    llm = ChatOpenAI(model=model_name, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    return llm_with_tools, tools, tool_map


def invoke_agent(llm_with_tools, messages: list) -> AIMessage:
    """Call the LLM with the current message history."""
    return llm_with_tools.invoke(messages)


def execute_tool_calls(tool_map: dict, tool_calls: list, tracer=None) -> list[ToolMessage]:
    """Execute tool calls with timeout and optional tracing. Returns ToolMessages."""
    tool_messages = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_call_id = tc["id"]

        if tool_name in tool_map:
            result, duration_ms, success = execute_with_timeout(
                tool_map[tool_name].invoke, tool_args, tool_name
            )
            if tracer:
                tracer.log_tool_executed(tool_name, duration_ms, success)
        else:
            result = f"Unknown tool: {tool_name}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call_id)
        )
    return tool_messages


def create_denial_messages(tool_calls: list) -> list[ToolMessage]:
    """Create ToolMessages indicating tool calls were denied by the user."""
    messages = []
    for tc in tool_calls:
        messages.append(
            ToolMessage(
                content=(
                    f"Tool '{tc['name']}' was denied by the user. "
                    "Please answer the question without using this tool, "
                    "based on what you already know."
                ),
                tool_call_id=tc["id"],
            )
        )
    return messages


def format_chat_history(messages: list[dict]) -> list:
    """Convert Streamlit message dicts to LangChain message objects."""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def build_initial_messages(user_input: str, chat_history: list) -> list:
    """Build the initial message list for an agent invocation."""
    messages = [SystemMessage(content=HARDENED_SYSTEM_PROMPT)]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_input))
    return messages
