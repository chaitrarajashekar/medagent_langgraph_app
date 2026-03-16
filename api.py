"""
MedAgent FastAPI — REST API wrapping the LangGraph pipeline.
Endpoints:
  POST /check-interaction  — main drug interaction check
  POST /chat               — chat-style input (same as LangFlow Chat Input)
  GET  /health             — health check
  GET  /                   — info
"""
import os, uuid

# ── Step 1: load .env but DO NOT override vars already set in shell ───
from dotenv import load_dotenv
load_dotenv(override=False)   # PowerShell $env: vars take priority over .env

# ── Step 2: Set up LangWatch ONCE — never call setup() again anywhere ─
import langwatch

_lw_key = os.getenv("LANGWATCH_API_KEY", "")
if _lw_key:
    langwatch.setup(api_key=_lw_key)
    print(f"✅ LangWatch ready — key loaded")
else:
    print("⚠️ LANGWATCH_API_KEY not set — traces will not appear in LangWatch")

# ── Step 3: Now safe to import graph ──────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from graph import medagent_graph
from state import MedAgentState
from nodes import parse_input_message


app = FastAPI(
    title="MedAgent AI",
    description=(
        "Drug Interaction Checker — LangGraph + LangSmith\n\n"
        "Every request is traced in LangSmith with full span breakdown:\n"
        "InputGuardrails → Agent1_RAGReader → Agent2_FDAValidator → ReportCompiler"
    ),
    version="4.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                  allow_methods=["*"], allow_headers=["*"])


# ── Request / Response models ─────────────────────────────────────────
class InteractionRequest(BaseModel):
    medications:        str = Field(..., example="warfarin:5:daily\naspirin:100:daily")
    patient_age:        str = Field(..., example="72")
    patient_conditions: str = Field("", example="atrial fibrillation, hypertension")
    clinical_question:  str = Field("", example="Is it safe to continue both?")

    class Config:
        json_schema_extra = {"example": {
            "medications":        "warfarin:5:daily\naspirin:100:daily",
            "patient_age":        "72",
            "patient_conditions": "atrial fibrillation, hypertension",
            "clinical_question":  "Is it safe to continue both medications?",
        }}


class ChatRequest(BaseModel):
    message: str = Field(..., example=(
        "medications: warfarin:5:daily\naspirin:100:daily\n"
        "age: 72\nconditions: atrial fibrillation\n"
        "question: Is it safe to continue both?"
    ))


class InteractionResponse(BaseModel):
    run_id:        str
    report:        str
    guardrail_ok:  bool
    langwatch_url: str


# ── Helper: run the graph ────────────────────────────────────────────
def run_pipeline(initial_state):
    run_id = str(uuid.uuid4())
    initial_state["run_id"]   = run_id
    initial_state["messages"] = []

    with langwatch.trace(name="MedAgentPipeline") as t:
        t.update(
            input=f"Patient {initial_state.get('patient_age','')}y | "
                  f"Meds: {initial_state.get('medications','')}",
        )
        result = medagent_graph.invoke(initial_state)
        t.update(
            output=result.get("final_report","")[:300],
        )

    return result, run_id

# ── Endpoints ─────────────────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "service":          "MedAgent AI v4",
        "stack":            "LangGraph + LangWatch",
        "docs":             "/docs",
        "health":           "/health",
        "langwatch":        "https://app.langwatch.ai",
        "tracing_active":   bool(os.getenv("LANGWATCH_API_KEY")),
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    key_loaded = bool(os.getenv("OPENAI_API_KEY"))
    lw_key     = bool(os.getenv("LANGWATCH_API_KEY"))
    return {
        "status":           "healthy" if key_loaded else "missing OPENAI_API_KEY",
        "openai_key":       "loaded" if key_loaded else "MISSING — add to .env",
        "langwatch_key":    "loaded" if lw_key else "not set — add LANGWATCH_API_KEY to .env",
        "langwatch_url":    "https://app.langwatch.ai",
        "tracing_active":   lw_key,
    }


@app.post("/check-interaction", response_model=InteractionResponse)
def check_interaction(req: InteractionRequest):
    """Structured input — medications, age, conditions, question."""

    # Guard: check key is available before running pipeline
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail=(
                "OPENAI_API_KEY not set. "
                "Add it to your .env file or uncomment the "
                "os.environ line in api.py"
            )
        )

    initial: MedAgentState = {
        "input_text":         "",
        "medications":        req.medications,
        "patient_age":        req.patient_age,
        "patient_conditions": req.patient_conditions,
        "clinical_question":  req.clinical_question,
        "guardrail_passed":   True,
        "guardrail_errors":   [],
        "rag_results":        [],
        "fda_results":        [],
        "final_report":       "",
        "run_id":             "",
        "messages":           [],
    }
    try:
        result, run_id = run_pipeline(initial)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    project = os.getenv("LANGCHAIN_PROJECT", "medagent-production")
    ls_url  = "https://app.langwatch.ai"

    return InteractionResponse(
        run_id=run_id,
        report=result.get("final_report", "No report generated."),
        guardrail_ok=result.get("guardrail_passed", True),
        langwatch_url=ls_url,
    )


@app.post("/chat", response_model=InteractionResponse)
def chat(req: ChatRequest):
    """Free-text chat input — mirrors LangFlow Chat Input node."""

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set. Add it to your .env file."
        )

    parsed   = parse_input_message(req.message)
    initial: MedAgentState = {
        "input_text":         req.message,
        "medications":        parsed["medications"],
        "patient_age":        parsed["patient_age"],
        "patient_conditions": parsed["patient_conditions"],
        "clinical_question":  parsed["clinical_question"],
        "guardrail_passed":   True,
        "guardrail_errors":   [],
        "rag_results":        [],
        "fda_results":        [],
        "final_report":       "",
        "run_id":             "",
        "messages":           [],
    }
    try:
        result, run_id = run_pipeline(initial)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    project = os.getenv("LANGCHAIN_PROJECT", "medagent-production")
    ls_url  = "https://app.langwatch.ai"

    return InteractionResponse(
        run_id=run_id,
        report=result.get("final_report", "No report generated."),
        guardrail_ok=result.get("guardrail_passed", True),
        langwatch_url=ls_url,
    )