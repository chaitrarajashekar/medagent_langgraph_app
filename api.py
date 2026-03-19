"""
MedAgent FastAPI — REST API wrapping the LangGraph pipeline.
Endpoints:
  POST /check-interaction  — main drug interaction check
  POST /chat               — chat-style input (same as LangFlow Chat Input)
  GET  /health             — health check
  GET  /                   — info
"""
import os, uuid, asyncio
from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=2)

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

# Run pipeline in a thread so we can return a proper error if it times out
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import asyncio
_executor = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
async def startup_event():
    """
    Pre-warm ChromaDB and embedding model on startup.
    This prevents the first request from timing out on Render.
    """
    print("🔥 Pre-warming ChromaDB and embedding model...")
    try:
        from nodes import get_chromadb
        get_chromadb()
        print("✅ ChromaDB ready")
    except Exception as e:
        print(f"⚠️  ChromaDB pre-warm failed: {e}")


# ── Request / Response models ─────────────────────────────────────────
class RegulatoryAuthority(str):
    """Supported regulatory authorities for Agent 2 validation."""
    FDA  = "FDA"    # US — OpenFDA adverse events + drug labels
    EMA  = "EMA"    # EU — EudraVigilance + CHMP decisions
    MHRA = "MHRA"   # UK — Yellow Card scheme + Drug Safety Updates

VALID_AUTHORITIES = {"FDA", "EMA", "MHRA"}
DEFAULT_AUTHORITIES = ["FDA"]   # FDA only by default — fastest single call


class InteractionRequest(BaseModel):
    medications:            str       = Field(...,  example="warfarin:5:daily\naspirin:100:daily")
    patient_age:            str       = Field(...,  example="72")
    patient_conditions:     str       = Field("",   example="atrial fibrillation, hypertension")
    clinical_question:      str       = Field("",   example="Is it safe to continue both?")
    regulatory_authorities: List[str] = Field(
        default=["FDA"],
        description=(
            "Which regulatory authorities Agent 2 should call. "
            "Options: FDA (US), EMA (EU), MHRA (UK). "
            "More authorities = stronger validation but higher latency. "
            "All selected authorities run in parallel."
        ),
        example=["FDA", "EMA", "MHRA"]
    )

    class Config:
        json_schema_extra = {"example": {
            "medications":            "warfarin:5:daily\naspirin:100:daily",
            "patient_age":            "72",
            "patient_conditions":     "atrial fibrillation, hypertension",
            "clinical_question":      "Is it safe to continue both medications?",
            "regulatory_authorities": ["FDA"],
        }}

    def get_authorities(self) -> List[str]:
        """Validate and return cleaned authority list."""
        chosen = [a.upper().strip() for a in (self.regulatory_authorities or ["FDA"])]
        invalid = [a for a in chosen if a not in VALID_AUTHORITIES]
        if invalid:
            raise ValueError(f"Invalid authorities: {invalid}. Valid: {VALID_AUTHORITIES}")
        return chosen if chosen else ["FDA"]


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
    """
    Wraps the full LangGraph pipeline in a single LangWatch trace.
    Uses context manager pattern (with langwatch.trace()) — more reliable
    than nested @langwatch.trace decorator which caused span context issues.
    """
    from langchain_core.runnables import RunnableConfig

    run_id = str(uuid.uuid4())
    initial_state["run_id"]   = run_id
    initial_state["messages"] = []

    lw_enabled = bool(os.getenv("LANGWATCH_API_KEY"))

    if not lw_enabled:
        # LangWatch not configured — run pipeline without tracing
        result = medagent_graph.invoke(initial_state)
        return result, run_id

    # Use context manager — creates parent trace that all node spans attach to
    with langwatch.trace(
        name="MedAgentPipeline",
        metadata={
            "medications":  initial_state.get("medications",""),
            "patient_age":  initial_state.get("patient_age",""),
            "conditions":   initial_state.get("patient_conditions",""),
            "authorities":  str(initial_state.get("regulatory_authorities",["FDA"])),
        }
    ) as trace:
        try:
            trace.update(
                input=(
                    f"Patient {initial_state.get('patient_age','')}y | "
                    f"{initial_state.get('patient_conditions','')} | "
                    f"Meds: {initial_state.get('medications','')}"
                )
            )
        except Exception:
            pass

        # Pass LangWatch callback via RunnableConfig so LLM calls
        # appear as named child spans inside this trace
        try:
            lw_callback = trace.get_langchain_callback()
            config = RunnableConfig(callbacks=[lw_callback])
        except Exception:
            config = RunnableConfig()

        result = medagent_graph.invoke(initial_state, config=config)

        # Update trace output and log evaluations
        try:
            trace.update(output=result.get("final_report","")[:500])

            trace.add_evaluation(
                name="Pipeline: Input Guardrail",
                passed=result.get("guardrail_passed", True),
                is_guardrail=True,
                details=(
                    "Input validation passed"
                    if result.get("guardrail_passed")
                    else str(result.get("guardrail_errors",[]))
                ),
            )

            fda_results = result.get("fda_results", [])
            if fda_results and not fda_results[0].get("blocked"):
                sev_score = {"MAJOR":0.0,"MODERATE":0.5,"MINOR":1.0,"UNKNOWN":0.5}
                final_sev = fda_results[0].get("final_severity","UNKNOWN")
                trace.add_evaluation(
                    name="Pipeline: Final Severity",
                    passed=True,
                    score=sev_score.get(final_sev, 0.5),
                    label=final_sev,
                    details=f"Agent2: {fda_results[0].get('agent2_decision','CONFIRM')}",
                )
        except Exception:
            pass

    return result, run_id

# ── Endpoints ─────────────────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"], operation_id="root_get")
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


@app.api_route("/health", methods=["GET", "HEAD"], operation_id="health_get")
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
async def check_interaction(req: InteractionRequest):
    """Structured input — medications, age, conditions, question."""

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500,
            detail="OPENAI_API_KEY not set. Add it to your .env file.")

    initial: MedAgentState = {
        "input_text":         "",
        "medications":        req.medications,
        "patient_age":        req.patient_age,
        "patient_conditions": req.patient_conditions,
        "clinical_question":       req.clinical_question,
        "regulatory_authorities":  getattr(req, "get_authorities", lambda: ["FDA"])(),
        "guardrail_passed":        True,
        "guardrail_errors":   [],
        "rag_results":        [],
        "fda_results":        [],
        "final_report":       "",
        "run_id":             "",
        "messages":           [],
    }
    try:
        # Run in thread pool so async event loop stays responsive
        loop   = asyncio.get_event_loop()
        result, run_id = await loop.run_in_executor(
            _executor, lambda: run_pipeline(initial)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    return InteractionResponse(
        run_id=run_id,
        report=result.get("final_report", "No report generated."),
        guardrail_ok=result.get("guardrail_passed", True),
        langwatch_url="https://app.langwatch.ai",
    )


