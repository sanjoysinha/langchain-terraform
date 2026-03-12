import os
import json
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from ingest import ingest_documents, load_vectorstore
from agent import (
    build_agent,
    invoke_agent,
    execute_tool_calls,
    create_denial_messages,
    format_chat_history,
    build_initial_messages,
)
from config import ROLES, DEFAULT_ROLE, MAX_AGENT_ITERATIONS
from guardrails import (
    validate_user_input,
    detect_prompt_injection,
    sanitize_output,
    ReadOnlyVectorstore,
    AgentTracer,
)

load_dotenv()

st.set_page_config(page_title="Document Chat - Agent", layout="wide")

TOOL_DISPLAY_NAMES = {
    "document_search": "Document Search",
    "document_summarize": "Document Summarize",
    "web_search": "Web Search",
}


# --- Session state initialization ---
def init_session_state():
    defaults = {
        "messages": [],
        "agent_messages": [],
        "agent_state": "idle",
        "pending_tool_calls": [],
        "pending_ai_message": None,
        "vectorstore": None,
        "llm_with_tools": None,
        "tool_map": None,
        "model_name": "gpt-4o",
        "role": DEFAULT_ROLE,
        "iteration_count": 0,
        "tracer": None,
        "trace_log": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()

# Auto-load persisted vectorstore on first run
if st.session_state.vectorstore is None:
    vs = load_vectorstore()
    if vs is not None:
        st.session_state.vectorstore = ReadOnlyVectorstore(vs)
        llm, tools, tmap = build_agent(
            st.session_state.vectorstore,
            st.session_state.model_name,
            role=st.session_state.role,
        )
        st.session_state.llm_with_tools = llm
        st.session_state.tool_map = tmap


# --- Sidebar ---
st.sidebar.header("Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

model_name = st.sidebar.selectbox(
    "Model",
    options=["gpt-4o", "gpt-3.5-turbo"],
    index=0,
)

role = st.sidebar.selectbox(
    "Role",
    options=list(ROLES.keys()),
    format_func=lambda r: ROLES[r]["label"],
    index=list(ROLES.keys()).index(DEFAULT_ROLE),
)

# Rebuild agent if model or role changed
if model_name != st.session_state.model_name or role != st.session_state.role:
    st.session_state.model_name = model_name
    st.session_state.role = role
    if st.session_state.vectorstore is not None:
        llm, tools, tmap = build_agent(
            st.session_state.vectorstore, model_name, role=role
        )
        st.session_state.llm_with_tools = llm
        st.session_state.tool_map = tmap

if st.sidebar.button("Process Documents", disabled=not uploaded_files):
    tmp_paths = []
    try:
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.close()
            tmp_paths.append(tmp.name)

        with st.spinner("Processing documents..."):
            raw_vs = ingest_documents(tmp_paths)
            st.session_state.vectorstore = ReadOnlyVectorstore(raw_vs)
            llm, tools, tmap = build_agent(
                st.session_state.vectorstore, model_name, role=role
            )
            st.session_state.llm_with_tools = llm
            st.session_state.tool_map = tmap
            st.session_state.messages = []
            st.session_state.agent_state = "idle"

        st.sidebar.success(f"Processed {len(uploaded_files)} document(s)!")
    finally:
        for path in tmp_paths:
            os.unlink(path)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.agent_state = "idle"
    st.session_state.agent_messages = []
    st.session_state.pending_tool_calls = []
    st.session_state.pending_ai_message = None
    st.session_state.iteration_count = 0
    st.session_state.tracer = None
    st.session_state.trace_log = []
    st.rerun()

# Sidebar status
if st.session_state.vectorstore is not None:
    st.sidebar.info("Vector store connected (AstraDB)")
else:
    st.sidebar.warning("No documents loaded. Upload PDFs or TXT files to get started.")

# --- Trace log display ---
if st.session_state.trace_log:
    with st.sidebar.expander("Agent Trace Log", expanded=False):
        for event in st.session_state.trace_log:
            elapsed = event.get("elapsed_ms", 0)
            event_type = event.get("event", "unknown")
            details = {k: v for k, v in event.items()
                       if k not in ("timestamp", "elapsed_ms", "event")}
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            st.text(f"[{elapsed}ms] {event_type}: {detail_str}")


# --- Agent step helper ---
def run_agent_step():
    """Invoke the LLM and handle the response, with iteration limiting."""
    st.session_state.iteration_count += 1
    tracer = st.session_state.tracer

    if st.session_state.iteration_count > MAX_AGENT_ITERATIONS:
        if tracer:
            tracer.log_max_iterations_reached(st.session_state.iteration_count - 1)
        st.session_state.agent_messages.append(
            HumanMessage(content=(
                "You have reached the maximum number of tool calls. "
                "Please provide your best answer now based on the information "
                "you have gathered so far. Do not request any more tools."
            ))
        )

    if tracer:
        tracer.log_llm_invoked(st.session_state.iteration_count)

    ai_msg = invoke_agent(
        st.session_state.llm_with_tools,
        st.session_state.agent_messages,
    )
    st.session_state.agent_messages.append(ai_msg)

    # If iteration limit exceeded, force final answer
    if st.session_state.iteration_count > MAX_AGENT_ITERATIONS:
        answer = ai_msg.content or "I was unable to generate a response within the allowed number of steps."
        answer = sanitize_output(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.agent_state = "idle"
        if tracer:
            tracer.log_final_answer(len(answer))
            st.session_state.trace_log = tracer.get_events()
        st.session_state.agent_messages = []
        return

    if ai_msg.tool_calls:
        if tracer:
            for tc in ai_msg.tool_calls:
                tracer.log_tool_proposed(tc["name"], tc["args"])
        st.session_state.pending_tool_calls = ai_msg.tool_calls
        st.session_state.pending_ai_message = ai_msg
        st.session_state.agent_state = "pending_approval"
    else:
        answer = sanitize_output(ai_msg.content)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.agent_state = "idle"
        if tracer:
            tracer.log_final_answer(len(answer))
            st.session_state.trace_log = tracer.get_events()
        st.session_state.agent_messages = []


# --- Main chat area ---
st.title("Chat with your Documents")

# Display all past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if message["role"] == "assistant":
            content = sanitize_output(content)
        st.markdown(content)

# --- State: pending_approval ---
if st.session_state.agent_state == "pending_approval":
    with st.chat_message("assistant"):
        st.markdown("**I'd like to use the following tool(s):**")
        for tc in st.session_state.pending_tool_calls:
            display_name = TOOL_DISPLAY_NAMES.get(tc["name"], tc["name"])
            st.info(
                f"**Tool:** {display_name}\n\n"
                f"**Arguments:** `{json.dumps(tc['args'])}`"
            )

        st.caption(
            f"Agent iteration {st.session_state.iteration_count} of {MAX_AGENT_ITERATIONS}"
        )
        remaining = MAX_AGENT_ITERATIONS - st.session_state.iteration_count
        if remaining <= 1:
            st.warning("This is the last allowed tool call iteration.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve", type="primary", use_container_width=True):
                tracer = st.session_state.tracer
                if tracer:
                    for tc in st.session_state.pending_tool_calls:
                        tracer.log_tool_approved(tc["name"])

                tool_messages = execute_tool_calls(
                    st.session_state.tool_map,
                    st.session_state.pending_tool_calls,
                    tracer=tracer,
                )
                st.session_state.agent_messages.extend(tool_messages)
                st.session_state.pending_tool_calls = []
                st.session_state.pending_ai_message = None
                run_agent_step()
                st.rerun()

        with col2:
            if st.button("Deny", use_container_width=True):
                tracer = st.session_state.tracer
                if tracer:
                    for tc in st.session_state.pending_tool_calls:
                        tracer.log_tool_denied(tc["name"])

                denial_messages = create_denial_messages(
                    st.session_state.pending_tool_calls,
                )
                st.session_state.agent_messages.extend(denial_messages)
                st.session_state.pending_tool_calls = []
                st.session_state.pending_ai_message = None
                run_agent_step()
                st.rerun()

# --- State: idle — accept new input ---
if st.session_state.agent_state == "idle":
    if prompt := st.chat_input("Ask a question about your documents"):
        if st.session_state.llm_with_tools is None:
            st.warning("Please upload and process documents first.")
        else:
            # Validate input
            is_valid, error_msg = validate_user_input(prompt)
            if not is_valid:
                st.error(error_msg)
            else:
                # Check for prompt injection
                is_suspicious, patterns = detect_prompt_injection(prompt)
                if is_suspicious:
                    st.warning(
                        "Your input contains patterns that resemble prompt injection. "
                        "The query will still be processed, but attempts to override "
                        "system instructions will not work."
                    )

                # Initialize tracer for this turn
                tracer = AgentTracer()
                tracer.log_user_input(prompt)
                if is_suspicious:
                    tracer.log_injection_warning(patterns)
                st.session_state.tracer = tracer
                st.session_state.iteration_count = 0

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                chat_history = format_chat_history(st.session_state.messages[:-1])
                st.session_state.agent_messages = build_initial_messages(
                    prompt, chat_history
                )

                with st.spinner("Thinking..."):
                    run_agent_step()

                st.rerun()
