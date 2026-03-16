"""
MedAgent LangGraph — builds and compiles the graph.
Flow matches the working LangFlow JSON exactly:
  ChatInput → Agent1 → Agent2 → ReportCompiler → ChatOutput
"""
from langgraph.graph import StateGraph, END
from state import MedAgentState
from nodes import (
    node_input_guardrails,
    node_agent1_rag,
    node_agent2_fda,
    node_report_compiler,
)


def should_continue_after_guardrails(state: MedAgentState) -> str:
    """
    Conditional edge after input guardrails.
    If guardrails blocked — skip to END (final_report already set).
    If passed — continue to Agent 1.
    """
    if not state.get("guardrail_passed", True):
        return "blocked"
    return "continue"


def should_continue_after_agent1(state: MedAgentState) -> str:
    """
    Conditional edge after Agent 1.
    If RxNorm blocked — skip Agent 2, go straight to Report Compiler.
    If normal — go to Agent 2.
    """
    rag = state.get("rag_results", [])
    if rag and rag[0].get("blocked"):
        return "blocked"
    return "continue"


def build_graph():
    graph = StateGraph(MedAgentState)

    # ── Add nodes ─────────────────────────────────────────────────────
    graph.add_node("input_guardrails",  node_input_guardrails)
    graph.add_node("agent1_rag",        node_agent1_rag)
    graph.add_node("agent2_fda",        node_agent2_fda)
    graph.add_node("report_compiler",   node_report_compiler)

    # ── Entry point ───────────────────────────────────────────────────
    graph.set_entry_point("input_guardrails")

    # ── Edges ─────────────────────────────────────────────────────────
    # After guardrails: blocked → END, passed → Agent 1
    graph.add_conditional_edges(
        "input_guardrails",
        should_continue_after_guardrails,
        {"blocked": END, "continue": "agent1_rag"},
    )

    # After Agent 1: RxNorm blocked → Report Compiler, normal → Agent 2
    graph.add_conditional_edges(
        "agent1_rag",
        should_continue_after_agent1,
        {"blocked": "report_compiler", "continue": "agent2_fda"},
    )

    # Agent 2 → Report Compiler → END
    graph.add_edge("agent2_fda",      "report_compiler")
    graph.add_edge("report_compiler", END)

    return graph.compile()


# Compiled graph — import this in api.py and run.py
medagent_graph = build_graph()
