import os
import time
from typing import TypedDict, List
from pydantic import BaseModel, Field
from settings import settings

# --- 0. Environment Setup ---
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT

from langgraph.graph import StateGraph, END
from local_oracle import get_retriever
from langchain_groq import ChatGroq


# --- 1. Define the State & Schema ---
class AgentState(TypedDict):
    input: str
    decision: str
    search_params: dict
    context: List[str]
    summaries: List[str]
    response: str
    safety_report: str
    docs_available: List[str]
    intermediate_steps: List[str]


class SearchPlan(BaseModel):
    scope: str = Field(
        description="The specific book name (e.g., 'Bug Bounty Bootcamp'). If broad/multiple, use 'GLOBAL'.")
    focus: str = Field(description="The technical vulnerability or tool to search for (e.g., 'SQLi').")
    decision: str = Field(
        description="Either 'research' for technical questions or 'clarify' for greetings/vague chat.")


# --- 2. Telemetry Decorator ---
def trace_node(func):
    def wrapper(state: AgentState):
        start_time = time.time()
        result = func(state)
        duration = time.time() - start_time
        if not isinstance(result, dict): result = {}
        # Ensure intermediate_steps is always initialized
        steps = state.get("intermediate_steps", []) + [f"Node {func.__name__} took {duration:.2f}s"]
        result["intermediate_steps"] = steps
        return result

    return wrapper


# Initialize base retriever once
base_retriever = get_retriever(force_reload=False)


# --- 3. The Nodes ---

@trace_node
def query_architect_node(state: AgentState):
    llm = ChatGroq(
        temperature=0,
        model_name=settings.FAST_MODEL,
        groq_api_key=settings.GROQ_API_KEY
    )
    structured_llm = llm.with_structured_output(SearchPlan)

    prompt = f"Analyze: '{state['input']}'. Available Docs: {state.get('docs_available', [])}. Identify SCOPE and FOCUS."
    plan = structured_llm.invoke(prompt)

    print(f"--- ARCHITECT: Scope={plan.scope} | Focus={plan.focus} | Decision={plan.decision} ---")
    return {
        "decision": plan.decision,
        "search_params": {"scope": plan.scope, "focus": plan.focus}
    }


@trace_node
def research_node(state: AgentState):
    scope = state['search_params']['scope']
    focus = state['search_params']['focus']
    query = focus if focus else state['input']

    # MASTER SEARCH: Pull 100 chunks for maximum coverage
    search_kwargs = {"k": 100, "fetch_k": 200, "lambda_mult": 0.5}

    # THE FIX: Use $contains for Chroma compatibility
    if scope and scope != "GLOBAL":
        search_kwargs["filter"] = {"source": {"$contains": scope}}
        print(f"--- SCOPED SEARCH: Isolating {scope} ---")
    else:
        print("--- GLOBAL SEARCH: No filters applied ---")

    base_retriever.search_kwargs = search_kwargs
    docs = base_retriever.invoke(query)

    context_parts = [f"--- SOURCE: {os.path.basename(d.metadata['source'])} ---\n{d.page_content}" for d in docs]
    return {"context": context_parts}


@trace_node
def summarizer_node(state: AgentState):
    llm = ChatGroq(
        temperature=0,
        model_name=settings.FAST_MODEL,
        groq_api_key=settings.GROQ_API_KEY
    )
    all_summaries = []
    batch_size = 5  # Meticulous batching

    context = state.get("context", [])
    if not context:
        return {"summaries": ["No relevant context found in documents."]}

    for i in range(0, len(context), batch_size):
        batch = context[i:i + batch_size]
        distillation_prompt = f"Extract EVERY technical phase, tool, and payload for '{state['input']}'. Do not omit details. CONTEXT: {batch}"
        summary = llm.invoke(distillation_prompt).content
        all_summaries.append(summary)
        print(f"--- Summarizer: Batch {i // batch_size + 1} processed. Sleeping 30s... ---")
        time.sleep(30)
    return {"summaries": all_summaries}


@trace_node
def responder_node(state: AgentState):
    llm = ChatGroq(
        temperature=0,
        model_name=settings.QUALITY_MODEL,
        groq_api_key=settings.GROQ_API_KEY
    )

    # We add a check to handle Guardrail rejections gracefully
    if state.get('decision') == 'reject':
        return {"response": "I'm sorry, but I can only assist with technical cybersecurity research and methodologies."}

    if state.get('decision') == 'research' and state.get('summaries'):
        combined_summaries = "\n\n".join(state['summaries'])
        synthesis_prompt = f"""SYSTEM: Master Technical Writer. 
        TASK: Construct a 'Master Answer' from the summaries. 
        SUMMARIES: {combined_summaries}
        QUESTION: {state['input']}"""
        response = llm.invoke(synthesis_prompt).content
    else:
        # This handles the case where the Architect says 'clarify'
        response = "I'm ready. Please provide a specific technical topic or book name to begin master research."

    return {"response": response}


@trace_node
def input_guardrail_node(state: AgentState):
    llm = ChatGroq(
        temperature=0,
        model_name=settings.FAST_MODEL,
        groq_api_key=settings.GROQ_API_KEY
    )

    # We give the Guardrail more context so it doesn't reject book-based research
    prompt = f"""
    SYSTEM: Security Research Gatekeeper.
    TASK: Classify the user input as 'SAFE' or 'REJECT'.

    SAFE CRITERIA:
    - Technical questions about hacking, web vulnerabilities (SQLi, XSS, etc.).
    - Questions about security tools (Burp, sqlmap).
    - **Researching specific security books or authors (e.g., Vickie Li, Bug Bounty Bootcamp).**

    User Input: "{state['input']}"

    Respond with ONLY 'SAFE' or 'REJECT'.
    """
    result = llm.invoke(prompt).content.strip().upper()
    print(f"--- DEBUG: Input Guardrail Result: {result} ---")

    return {"decision": "research" if "SAFE" in result else "reject"}


@trace_node
def output_audit_node(state: AgentState):
    llm = ChatGroq(
        temperature=0,
        model_name=settings.FAST_MODEL,
        groq_api_key=settings.GROQ_API_KEY
    )

    # Sample summaries to stay under TPM limits
    summaries_preview = "\n".join(state.get('summaries', [])[:5])

    audit_prompt = f"""
    SYSTEM: Forensic Audit Specialist.
    TASK: Compare the MASTER ANSWER against the TECHNICAL SUMMARIES.

    If accurate and grounded, respond 'PASS'.
    If missing data or wrong, respond 'FAIL: [Reason]'.

    SUMMARIES (Preview): {summaries_preview}
    MASTER ANSWER: {state['response']}
    """
    audit_result = llm.invoke(audit_prompt).content.strip()

    if "PASS" in audit_result.upper() and len(audit_result) < 10:
        return {"response": state['response'], "safety_report": "PASS"}
    else:
        return {"response": f"{state['response']}\n\n--- 🔍 AUDIT FEEDBACK ---\n{audit_result}",
                "safety_report": "AUDITED"}


# --- 4. Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("query_architect", query_architect_node)
workflow.add_node("research", research_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("responder", responder_node)
workflow.add_node("output_audit", output_audit_node)

workflow.set_entry_point("input_guardrail")

# Edge logic
workflow.add_conditional_edges(
    "input_guardrail",
    lambda x: x["decision"],
    {"research": "query_architect", "reject": "responder"}
)
workflow.add_conditional_edges(
    "query_architect",
    lambda x: x["decision"],
    {"research": "research", "clarify": "responder"}
)

workflow.add_edge("research", "summarizer")
workflow.add_edge("summarizer", "responder")
workflow.add_edge("responder", "output_audit")
workflow.add_edge("output_audit", END)

app = workflow.compile()

# --- 5. Corrected Interactive Loop ---
if __name__ == "__main__":
    data_path = settings.DATA_PATH
    files = [f for f in os.listdir(data_path) if f.endswith('.pdf')] if os.path.exists(data_path) else []

    while True:
        user_question = input("\nUser: ")
        if user_question.lower() in ["exit", "quit"]: break

        # THE FIX: Initialize 'summaries' as an empty list to prevent KeyErrors
        initial_input = {
            "input": user_question,
            "docs_available": files,
            "intermediate_steps": [],
            "summaries": [],
            "context": []
        }

        print("\n--- Initiating Master Research (Est. 10-15 mins)... ---")
        final_state = app.invoke(initial_input)
        print(f"\nOracle: {final_state['response']}")