"""
Quick local test — runs the pipeline directly without FastAPI.
Usage: python run_local.py
"""
import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import uuid
from dotenv import load_dotenv
load_dotenv()

from graph import medagent_graph
from state import MedAgentState

test_cases = [
    {
        "name": "Warfarin + Aspirin (MAJOR — should pass all guardrails)",
        "medications": "warfarin:5:daily\naspirin:100:daily",
        "patient_age": "72",
        "patient_conditions": "atrial fibrillation, hypertension",
        "clinical_question": "Is it safe to continue both medications?",
    },
    {
        "name": "Fictional drugs (should be blocked by RxNorm)",
        "medications": "zanthoferol:100:daily\nlysopradine:50:daily",
        "patient_age": "60",
        "patient_conditions": "hypertension",
        "clinical_question": "Check interactions",
    },
    {
        "name": "Bad input — only one medication (should fail input guardrail)",
        "medications": "warfarin:5:daily",
        "patient_age": "72",
        "patient_conditions": "AF",
        "clinical_question": "Any issues?",
    },
]

for tc in test_cases:
    print(f"\n{'='*60}")
    print(f"TEST: {tc['name']}")
    print(f"{'='*60}")

    state: MedAgentState = {
        "input_text":         "",
        "medications":        tc["medications"],
        "patient_age":        tc["patient_age"],
        "patient_conditions": tc["patient_conditions"],
        "clinical_question":  tc["clinical_question"],
        "guardrail_passed":   True,
        "guardrail_errors":   [],
        "rag_results":        [],
        "fda_results":        [],
        "final_report":       "",
        "run_id":             str(uuid.uuid4()),
        "messages":           [],
    }

    result = medagent_graph.invoke(state)
    print("\n" + result.get("final_report","No report"))
    print(f"\nGuardrail passed: {result.get('guardrail_passed')}")
    print(f"Run ID (for LangSmith): {state['run_id']}")
