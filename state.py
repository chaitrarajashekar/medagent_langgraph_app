"""
MedAgent State — shared typed dict passed between all LangGraph nodes.
Matches exactly the data flow in the working LangFlow JSON:
  ChatInput → Agent1RAGReader → Agent2FDAValidator → ReportCompiler → ChatOutput
"""
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


class MedAgentState(TypedDict):
    # ── Raw inputs (from ChatInput or API) ────────────────────────────
    input_text:         str          # full user message from Chat Input
    medications:        str          # parsed: "warfarin:5:daily\naspirin:100:daily"
    patient_age:        str          # parsed: "72"
    patient_conditions: str          # parsed: "atrial fibrillation, hypertension"
    clinical_question:  str          # parsed: "Is it safe to continue both?"

    # ── Guardrail results ─────────────────────────────────────────────
    guardrail_passed:   bool
    guardrail_errors:   List[str]

    # ── Agent 1 output ────────────────────────────────────────────────
    rag_results:        List[dict]   # [{drug_a, drug_b, severity, summary, ...}]

    # ── Agent 2 output ────────────────────────────────────────────────
    fda_results:        List[dict]   # [{drug_a, drug_b, fda_events_a, validation, final_severity, ...}]

    # ── Final report ──────────────────────────────────────────────────
    final_report:       str          # plain text — what Chat Output displays

    # ── LangSmith metadata ────────────────────────────────────────────
    run_id:             str
    messages:           Annotated[list, add_messages]
