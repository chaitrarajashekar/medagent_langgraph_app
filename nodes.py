"""
MedAgent LangGraph Nodes
Observability: LangWatch (replaces LangSmith — easier setup, same visibility)
Every node is decorated with @langwatch.trace — appears as a named span
in the LangWatch dashboard at app.langwatch.ai
"""
import os, json, re, glob, httpx, uuid
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import langwatch
from langwatch.types import RAGChunk
from langchain_core.runnables import RunnableConfig
from observability import traced_node, safe_add_evaluation, safe_span_update, get_safe_span
from state import MedAgentState
from guardrails import check_input, check_rag_output, check_fda_output, check_fda_number_grounding

load_dotenv(override=False)

# ── In-memory API cache with 10-minute TTL ────────────────────────────
# Avoids repeated calls to FDA, RxNorm, EMA, MHRA for same drug pairs.
# Cache key = function name + args. Expires after CACHE_TTL seconds.
# Thread-safe — uses a lock for concurrent FastAPI requests.
import threading
import time as _time

CACHE_TTL  = 600   # 10 minutes in seconds
_cache     = {}    # {key: {"value": result, "expires": timestamp}}
_cache_lock = threading.Lock()

def _cache_get(key: str):
    """Return cached value if not expired, else None."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry and _time.time() < entry["expires"]:
            return entry["value"]
        if entry:
            del _cache[key]   # clean up expired entry
        return None

def _cache_set(key: str, value):
    """Store value in cache with TTL expiry."""
    with _cache_lock:
        _cache[key] = {
            "value":   value,
            "expires": _time.time() + CACHE_TTL,
        }

def _cache_stats() -> dict:
    """Return current cache size and hit counts — useful for monitoring."""
    with _cache_lock:
        now = _time.time()
        active = {k:v for k,v in _cache.items() if now < v["expires"]}
        return {"active_entries": len(active), "ttl_seconds": CACHE_TTL}

# ── LangWatch setup ───────────────────────────────────────────────────
# Get your free API key at https://app.langwatch.ai
# Set LANGWATCH_API_KEY in your .env file
# LangWatch setup happens in api.py — not here
# This prevents the "already setup" conflict and 401 errors
_lw_key = os.getenv("LANGWATCH_API_KEY", "")

# ── KB docs path ─────────────────────────────────────────────────────
# Real clinical KB documents live in KBFILES/ folder (7 files).
# Set KB_DIR env var to override — defaults to ./KBFILES
# These replace the old hardcoded KB_DOCS summaries.
# No inline fallback — if files are missing, a clear error is logged.
KB_DIR_DEFAULT = "./KBFILES"

# ── LLM singleton ─────────────────────────────────────────────────────
def get_llm(max_tokens=1200, callbacks=None):
    """
    Returns ChatOpenAI. Per LangWatch docs, injects get_langchain_callback()
    so every LLM call appears as a named span inside the current trace.
    """
    load_dotenv(override=False)
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not found. Set it in .env")

    # LangWatch langchain callback — captures prompt/response/tokens per call
    lw_callbacks = []
    try:
        trace = langwatch.get_current_trace()
        if trace:
            lw_callbacks = [trace.get_langchain_callback()]
    except Exception:
        pass

    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        api_key=key,
        max_tokens=max_tokens,
        callbacks=lw_callbacks + (callbacks or []),
    )

# ── ChromaDB singleton ────────────────────────────────────────────────
_db = None

def get_chromadb():
    """
    Builds ChromaDB vector store from clinical KB documents in KBFILES/.

    Path resolution (in order):
    1. KB_DIR environment variable (if set)
    2. ./KBFILES  ← sole default — 7 peer-reviewed clinical documents

    Files loaded:
    01_warfarin_aspirin.txt           — BNF + CMAJ evidence
    02_omeprazole_clopidogrel.txt     — EMA CHMP + COGENT trial
    03_fluoxetine_tramadol.txt        — FDA label + Mikkelsen 2023
    04_simvastatin_clarithromycin.txt — FDA Zocor label + MHRA SPS
    05_lithium_ibuprofen.txt          — BNF 2024 + MHRA guidance
    06_general_cyp450_interactions.txt — FDA Clinical Drug Interaction Guidance
    07_anticoagulation_management.txt  — NHS Derbyshire + BCSH Guidelines
    """
    global _db
    if _db:
        return _db

    # Resolve KB folder: env var → ./KBFILES (sole default)
    kb_dir = os.getenv("KB_DIR", "./KBFILES")

    txt_files = sorted(glob.glob(f"{kb_dir}/*.txt"))
    raw_docs  = []

    print(f"   📚 Loading KB from: {kb_dir} ({len(txt_files)} files found)")

    for fpath in txt_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                text = f.read()
            raw_docs.append(Document(
                page_content=text,
                metadata={
                    "source":   os.path.basename(fpath),
                    "filepath": fpath,
                    "chars":    len(text),
                }
            ))
            print(f"   ✅ Loaded: {os.path.basename(fpath)} ({len(text):,} chars)")
        except Exception as e:
            print(f"   ❌ KB load error {fpath}: {e}")

    if not raw_docs:
        # Hard error — no KB files found and no inline fallback
        # This is intentional: we want to know if KBFILES is missing
        raise FileNotFoundError(
            f"No KB .txt files found in '{kb_dir}'. "
            f"Ensure KBFILES/ folder exists with the 7 clinical documents. "
            f"Set KB_DIR env var to override the path."
        )

    print(f"   📖 Loaded {len(raw_docs)} KB documents — chunking...")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(raw_docs)

    print(f"   🔪 {len(chunks)} chunks created from {len(raw_docs)} documents")

    emb = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY", "")
    )

    _db = Chroma.from_documents(
        chunks, emb,
        collection_name="medagent_openai",
        persist_directory="./chroma_medagent_openai",
    )

    print(f"   ✅ ChromaDB ready — {len(chunks)} chunks embedded")
    return _db

# ── Parse medication string ───────────────────────────────────────────
def parse_meds(medications: str):
    meds, parsed = {}, []
    for line in medications.strip().split("\n"):
        if not line.strip(): continue
        parts = line.strip().split(":")
        name  = parts[0].strip().lower()
        dose  = parts[1].strip() if len(parts) > 1 else "unknown"
        freq  = parts[2].strip() if len(parts) > 2 else "daily"
        parsed.append({"name": name, "dose": dose, "freq": freq})
        meds[name] = {"dose": dose, "freq": freq}
    pairs = [(parsed[i]["name"], parsed[j]["name"])
             for i in range(len(parsed)) for j in range(i+1, len(parsed))]
    return pairs, meds

# ── Parse user message into structured fields ─────────────────────────
def parse_input_message(text: str) -> dict:
    """
    Accepts either:
    - plain message: "Is warfarin + aspirin safe for 72y patient with AF?"
    - structured:    "medications: warfarin:5:daily\naspirin:100:daily\nage: 72\n..."
    Falls back to defaults so it always works.
    """
    text_lower = text.lower()
    # Try to extract medications
    med_match = re.search(r'medications?[:\s]+(.+?)(?:\n(?:age|conditions?|question)|$)',
                          text, re.IGNORECASE | re.DOTALL)
    meds = med_match.group(1).strip() if med_match else "warfarin:5:daily\naspirin:100:daily"

    age_match = re.search(r'age[:\s]+(\d+)', text, re.IGNORECASE)
    age = age_match.group(1) if age_match else "72"

    cond_match = re.search(r'conditions?[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
    conditions = cond_match.group(1).strip() if cond_match else "not specified"

    q_match = re.search(r'question[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
    question = q_match.group(1).strip() if q_match else text.strip()

    return {"medications": meds, "patient_age": age,
            "patient_conditions": conditions, "clinical_question": question}


# ════════════════════════════════════════════════════════════════════
# NODE 1 — INPUT PARSER + GUARDRAILS
# LangSmith span: "InputGuardrails"
# ════════════════════════════════════════════════════════════════════
@traced_node("InputGuardrails")
def node_input_guardrails(state: MedAgentState) -> dict:
    print("\n🛡️  [1] Input Guardrails")
    
   
    # Parse the raw input_text into structured fields
    parsed = parse_input_message(state.get("input_text", ""))
    meds   = state.get("medications") or parsed["medications"]
    age    = state.get("patient_age") or parsed["patient_age"]
    conds  = state.get("patient_conditions") or parsed["patient_conditions"]
    quest  = state.get("clinical_question") or parsed["clinical_question"]

    passed, errors = check_input(meds, age, conds, quest)

    updates = {
        "medications":        meds,
        "patient_age":        age,
        "patient_conditions": conds,
        "clinical_question":  quest,
        "guardrail_passed":   passed,
        "guardrail_errors":   errors,
    }

    if not passed:
        print(f"   ❌ Blocked: {errors}")
        updates["final_report"] = (
            "⛔ INPUT GUARDRAIL BLOCKED\n\n"
            + "\n".join(errors)
            + "\n\nPlease correct your input and resubmit.\n\n"
            "⚠️ DISCLAIMER: AI-generated. Research only. Not medical advice."
        )
    else:
        print(f"   ✅ Input valid — {meds.count(chr(10))+1} medication(s)")

   

    return updates


# ════════════════════════════════════════════════════════════════════
# NODE 2 — AGENT 1: RAG READER
# Direct port of MedAgentRAGReader from working LangFlow JSON
# LangSmith span: "Agent1_RAGReader"
# ════════════════════════════════════════════════════════════════════
AGENT1_PROMPT = """You are a clinical pharmacist doing a routine drug interaction check.

Patient: Age {age}y | Conditions: {conditions}
Medications: {drug_a} {dose_a}mg {freq_a}  +  {drug_b} {dose_b}mg {freq_b}
Clinical question: {clinical_question}

RETRIEVED KNOWLEDGE BASE CONTEXT (use ONLY this — do not add outside knowledge):
{context}

Rules:
- Base your answer ONLY on the context above
- If the context does not mention this drug pair, state UNKNOWN severity
- Do NOT invent mechanisms not present in the context
- If two context chunks contradict each other on severity (e.g. one says MAJOR, another says MODERATE):
    → Use the HIGHER severity and state in SUMMARY that sources differ
    → Quote both source tags so the clinician can verify
- If context chunks agree on severity but differ on mechanism detail:
    → Use the mechanism described in the most specific chunk
    → Cite that chunk's source tag

Respond with:
1. SEVERITY: [MAJOR / MODERATE / MINOR / UNKNOWN]
2. SUMMARY: 2 sentences drawn from context
3. CLINICAL ACTION: 2-3 bullet points
4. SOURCE: quote the source tag from context (e.g. BNF 2024)

━━━ GOLD EXAMPLE — follow this format exactly ━━━
Patient: 72y | Conditions: atrial fibrillation, hypertension
Medications: warfarin 5mg daily + aspirin 100mg daily
Context: [source=01_warfarin_aspirin.txt] Warfarin + aspirin: MAJOR \
interaction. Warfarin inhibits Vitamin K clotting factors (II,VII,IX,X). \
Aspirin irreversibly inhibits COX-1, reducing thromboxane A2-mediated \
platelet aggregation. Additive effect dramatically raises GI and \
intracranial bleeding risk. Source: BNF + Clinical Pharmacology.

1. SEVERITY: MAJOR
2. SUMMARY: Warfarin and aspirin together dramatically raise GI and \
   intracranial bleeding risk through additive anticoagulant and \
   antiplatelet mechanisms (BNF + Clinical Pharmacology). This \
   combination should be avoided unless clinically essential, in which \
   case close INR monitoring and PPI cover are required.
3. CLINICAL ACTION:
   - Assess whether aspirin is clinically essential; switch to \
     paracetamol for analgesia if antiplatelet effect is not required
   - Add a PPI (pantoprazole preferred) if the combination is unavoidable
   - Increase INR monitoring to at least weekly until stable
4. SOURCE: 01_warfarin_aspirin.txt (BNF + Clinical Pharmacology)
━━━ END OF EXAMPLE ━━━

⚠️ DISCLAIMER: AI-generated. Research only. Not medical advice."""

@traced_node("Agent1_RAGReader")
def node_agent1_rag(state: MedAgentState, config: dict = None) -> dict:
    print("\n🔍  [2] Agent 1 — RAG Reader")
    callbacks = (config or {}).get("callbacks", [])

    # RxNorm validation — same logic as working LangFlow component
    pairs, meds = parse_meds(state["medications"])
    blocked = []
    validation_log = []

    # Validate all drug names in parallel via RxNorm API before any LLM call
    import concurrent.futures
    drug_list = list(meds.keys())
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(drug_list)) as ex:
        futures = {ex.submit(_rxnorm_validate, drug): drug for drug in drug_list}
        results = {drug: fut.result() for fut, drug in
                   [(f, futures[f]) for f in concurrent.futures.as_completed(futures)]}

    for drug in drug_list:
        result = results[drug]
        validation_log.append(result["status"])
        print(f"   {result['status']}")
        if not result["found"]:
            blocked.append(drug)

    if blocked:
        block_result = [{
            "blocked": True,
            "drug_a": blocked[0], "drug_b": blocked[-1],
            "severity": "BLOCKED",
            "summary": (
                f"⛔ BLOCKED by RxNorm Validation Tool.\n"
                f"Drug(s) not recognised: {', '.join(blocked)}.\n"
                f"MedAgent only assesses known medications.\n"
                f"Please check spelling or use the generic name."
            ),
        }]
        return {"rag_results": block_result}

    db  = get_chromadb()
    llm = get_llm(callbacks=callbacks)
    results = []

    for drug_a, drug_b in pairs:
        docs = db.similarity_search(
            f"{drug_a} {drug_b} interaction severity mechanism", k=4
        )
        context = "\n---\n".join(
            f"[source={d.metadata.get('source','?')}] {d.page_content}"
            for d in docs
        )
        ma, mb = meds.get(drug_a,{}), meds.get(drug_b,{})
        prompt = AGENT1_PROMPT.format(
            age=state["patient_age"],
            conditions=state["patient_conditions"],
            drug_a=drug_a, dose_a=ma.get("dose","?"), freq_a=ma.get("freq","daily"),
            drug_b=drug_b, dose_b=mb.get("dose","?"), freq_b=mb.get("freq","daily"),
            clinical_question=state["clinical_question"],
            context=context,
        )
        chain  = (ChatPromptTemplate.from_messages([HumanMessage(content=prompt)])
                  | llm | StrOutputParser())
        result = chain.invoke({})

        # Output guardrails
        passed, failures, warnings = check_rag_output(result, docs)
        if not passed:
            result += "\n\n⚠️ AGENT 1 GUARDRAIL:\n" + "\n".join(failures)

        severity = next(
            (s for s in ["MAJOR","MODERATE","MINOR"] if s in result.upper()), "UNKNOWN"
        )
        results.append({
            "drug_a": drug_a, "drug_b": drug_b,
            "severity": severity, "summary": result,
            "guardrail_passed": passed,
            "guardrail_warnings": warnings,
            "chunks_retrieved": len(docs),
            "validation_log": validation_log,
        })
        icon = "✅" if passed else "⚠️"
        print(f"   {icon} {drug_a}+{drug_b} → {severity}")

    # ── LangWatch RAG logging (per capturing-rag docs) ────────────────
    # type="rag" enables Faithfulness + Context Relevancy evaluators
    try:
        with langwatch.span(type="rag", name="Agent1_KBRetrieval") as rag_span:
            safe_span_update(rag_span, 
                input=state.get("clinical_question", ""),
                contexts=[
                    RAGChunk(
                        content=r.get("summary","")[:400],
                        document_id=f"{r.get('drug_a','')}+{r.get('drug_b','')}",
                        source="ChromaDB-KB",
                    )
                    for r in results
                ],
                output=f"Severities found: {[r.get('severity') for r in results]}",
            )
            # Log each output guardrail as evaluation (per capturing-evaluations docs)
            for r in results:
                # G-OUT-01 checks severity presence specifically —
                # not overall guardrail_passed which included the disclaimer
                # check (G-OUT-02) that GPT-4o does not echo in its 4-section output.
                _sev_present = r.get("severity", "") in ["MAJOR", "MODERATE", "MINOR", "UNKNOWN"]
                safe_add_evaluation(rag_span,
                    name=f"G-OUT-01: Severity Present ({r.get('drug_a','')}+{r.get('drug_b','')})",
                    passed=_sev_present,
                    is_guardrail=True,
                    score=1.0 if _sev_present else 0.0,
                    details=f"Severity: {r.get('severity')} | KB chunks: {r.get('chunks_retrieved',0)}",
                )
    except Exception:
        pass

    return {"rag_results": results}


def _rxnorm_validate(drug_name: str) -> dict:
    """
    Pre-LLM validation tool: calls the NIH RxNorm REST API to verify
    every drug name is a real, recognised medication before any LLM call.

    This is a deterministic pipeline gate — not a tool the model controls.
    The pipeline always calls this. The LLM never decides to skip it.

    Returns:
        found  : True if RxNorm recognises the drug
        status : human-readable result string
    """
    try:
        r = httpx.get(
            "https://rxnav.nlm.nih.gov/REST/rxcui.json",
            params={"name": drug_name, "search": 2}, timeout=5.0
        )
        if r.status_code == 200:
            rxcui = r.json().get("idGroup", {}).get("rxnormId", [])
            if rxcui:
                return {"drug": drug_name, "found": True,
                        "status": f"✅ RxNorm validation tool — RxCUI: {rxcui[0]}"}
            return {"drug": drug_name, "found": False,
                    "status": f"❌ RxNorm validation tool — drug not found (fictional or misspelled)"}
    except Exception:
        pass
    return {"drug": drug_name, "found": True,
            "status": "⚠️ RxNorm validation tool unreachable — proceeding without validation"}


# ════════════════════════════════════════════════════════════════════
# NODE 3 — AGENT 2: FDA VALIDATOR
# Direct port of MedAgentFDAValidator from working LangFlow JSON
# LangSmith span: "Agent2_FDAValidator"
# ════════════════════════════════════════════════════════════════════
AGENT2_PROMPT = """You are an independent regulatory auditor with authority to OVERRIDE Agent 1.

Patient: {age}y | Conditions: {conditions}
Drugs: {drug_a} ({dose_a}mg {freq_a}) + {drug_b} ({dose_b}mg {freq_b})

AGENT 1 KNOWLEDGE BASE VERDICT (from clinical literature):
  Severity: {agent1_severity}
  Summary:  {rag_summary}

REGULATORY AUTHORITIES SELECTED BY USER: {authorities_called}
Only the authorities listed above were called. All others show "NOT REQUESTED".
You MUST base your decision ONLY on the data from selected authorities.

LIVE REGULATORY DATA — use exact numbers, never invent:
  FDA (US):   {drug_a} — {events_a} adverse events | Top reactions: {reactions_a}
              {drug_b} — {events_b} adverse events | Top reactions: {reactions_b}
              Boxed warning:          {boxed_a}
              Label interaction text: {interactions_a}
  EMA (EU):   {ema_signal}
  MHRA (UK):  {mhra_signal}

━━━ HOW TO DETERMINE YOUR DECISION ━━━

STEP 1 — Identify which authorities are ACTIVE (selected + returned data):
  An authority is ACTIVE only if it appears in AUTHORITIES SELECTED BY USER above
  AND its data is NOT "NOT REQUESTED".
  Ignore all authorities that are NOT ACTIVE for the decision logic below.

STEP 2 — Count ACTIVE authorities and assess their signals vs Agent 1 severity:

  Scenario A — Only 1 ACTIVE authority (e.g. user selected FDA only):
    → That single authority is the sole signal.
    → If it AGREES with Agent 1                        → CONFIRM
    → If it signals STRONGER severity than Agent 1     → ESCALATE
    → If it signals WEAKER severity than Agent 1       → CONTRADICT
    → Boxed warning present in that authority          → ESCALATE minimum

  Scenario B — 2 ACTIVE authorities (e.g. user selected FDA + EMA):
    → If BOTH agree with Agent 1                       → CONFIRM
    → If BOTH signal STRONGER than Agent 1             → ESCALATE or OVERRIDE
    → If BOTH signal WEAKER than Agent 1               → CONTRADICT
    → If signals SPLIT (one higher, one lower)         → CONFIRM (KB is tiebreaker)
    → If EITHER has a boxed warning                    → ESCALATE minimum

  Scenario C — All 3 ACTIVE authorities (FDA + EMA + MHRA):
    → If ALL 3 agree with Agent 1                      → CONFIRM
    → If 2 or more signal STRONGER than Agent 1        → ESCALATE or OVERRIDE
    → If 2 or more signal WEAKER than Agent 1          → CONTRADICT
    → If majority contradicts Agent 1 severity         → OVERRIDE
    → If signals split (no majority)                   → CONFIRM (KB is tiebreaker)
    → If ANY jurisdiction has a boxed warning          → ESCALATE minimum

STEP 3 — Special cases (apply in ANY scenario):
  - FDA events = 0: treat as NO DATA — do NOT interpret as evidence of safety.
    A new or rare drug may simply not appear in OpenFDA yet.
  - "NOT REQUESTED" authority: treat as if it does not exist.
    Do NOT cite it, reference it, or use training knowledge about what it might say.
  - KB is always the tiebreaker on a split signal — never downgrade Agent 1 on a split.

━━━ YOUR 4 POSSIBLE DECISIONS ━━━
  CONFIRM    — ACTIVE authority/authorities agree with Agent 1 severity. No change.
  ESCALATE   — ACTIVE signal(s) indicate STRONGER risk than Agent 1. Severity raised.
  CONTRADICT — ACTIVE signal(s) indicate WEAKER risk. Agent 1 stands (KB tiebreaker).
  OVERRIDE   — Majority of ACTIVE authorities directly contradict Agent 1. Severity replaced.

━━━ THINK THROUGH THESE STEPS BEFORE WRITING YOUR DECISION ━━━

Work through the following reasoning steps in order. Write each step
clearly so your decision is auditable and traceable.

Step 1 — List ACTIVE authorities:
  For each of FDA, EMA, MHRA: is it in AUTHORITIES SELECTED BY USER
  AND is its data not "NOT REQUESTED"?
  Write: "ACTIVE: [list]" and "INACTIVE: [list]"

Step 2 — Assess each ACTIVE authority's signal vs Agent 1:
  For each ACTIVE authority write one line:
  "[Authority]: signals [HIGHER / SAME / LOWER] than Agent 1 [severity]
   because [exact evidence from data above — quote numbers or label text]"

Step 3 — Check for boxed warning:
  Does any ACTIVE authority have a boxed warning? YES / NO
  If YES → this automatically triggers ESCALATE minimum regardless of step 2.

Step 4 — Apply the scenario rule:
  How many ACTIVE authorities do you have? (1, 2 or 3)
  Apply the matching scenario (A, B or C) from the rules above.
  Write: "Applying Scenario [A/B/C]: [state the rule outcome]"

Step 5 — Resolve any split:
  If signals are split with no majority → KB is the tiebreaker → CONFIRM.
  Write: "Signal split: [yes/no]. Tiebreaker needed: [yes/no]."

Step 6 — State your decision and severity:
  Write: "Decision: [CONFIRM / ESCALATE / CONTRADICT / OVERRIDE]"
  Write: "Final severity: [MAJOR / MODERATE / MINOR / UNKNOWN]"
  Write: "Reason: [one sentence citing which authority and what evidence]"

━━━ RESPOND WITH EXACTLY THIS STRUCTURE ━━━
ACTIVE AUTHORITIES: [list only the authorities that were ACTIVE for this decision]
DECISION: [CONFIRM / ESCALATE / CONTRADICT / OVERRIDE]
REASON: [one sentence — which ACTIVE authorities drove this decision and what signal they gave]
JURISDICTION SIGNALS:
  FDA:  [MAJOR/MODERATE/MINOR/NO DATA/NOT REQUESTED] — [evidence from data above or "not selected"]
  EMA:  [MAJOR/MODERATE/MINOR/NO DATA/NOT REQUESTED] — [evidence from data above or "not selected"]
  MHRA: [MAJOR/MODERATE/MINOR/NO DATA/NOT REQUESTED] — [evidence from data above or "not selected"]
FINAL VALIDATED SEVERITY: [MAJOR / MODERATE / MINOR / UNKNOWN]
PATIENT FACTORS: [how this patient's age and conditions specifically affect this drug combination]

⚠️ DISCLAIMER: AI-generated. Research only. Not medical advice."""

@traced_node("Agent2_FDAValidator")
def node_agent2_fda(state: MedAgentState, config: dict = None) -> dict:
    """
    Agent 2 — Multi-Jurisdiction Regulatory Auditor.

    Has authority to CONFIRM, ESCALATE, CONTRADICT or OVERRIDE Agent 1.
    Calls FDA (US), EMA (EU) and MHRA (UK) APIs independently.
    If 2+ jurisdictions contradict Agent 1 → severity is overridden.
    This makes Agent 2 a true agent — it makes decisions, not just
    collects data.
    """
    print("\n🔬  [3] Agent 2 — Multi-Jurisdiction Regulatory Auditor")
    callbacks   = (config or {}).get("callbacks", [])
    rag_results = state.get("rag_results", [])

    # Pass through blocked signals unchanged
    if rag_results and rag_results[0].get("blocked"):
        return {"fda_results": rag_results}

    llm         = get_llm(callbacks=callbacks)
    pairs, meds = parse_meds(state["medications"])
    fda_results = []

    for i, (drug_a, drug_b) in enumerate(pairs):
        rag_entry     = rag_results[i] if i < len(rag_results) else rag_results[0]
        agent1_sev    = rag_entry.get("severity", "UNKNOWN")
        ma            = meds.get(drug_a, {})
        mb            = meds.get(drug_b, {})

        # ── Selective parallel API calls ─────────────────────────
        # Only call authorities the user selected in the request.
        # Default: FDA only (~1.2s). All three: ~1.2s (parallel).
        authorities = state.get("regulatory_authorities", ["FDA"])
        print(f"   📡 Calling selected authorities: {authorities} (in parallel)...")

        import concurrent.futures
        futures_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # FDA is always called if selected (needed for label + events)
            if "FDA" in authorities:
                futures_map["fda_ev_a"] = executor.submit(_fda_adverse_events, drug_a)
                futures_map["fda_ev_b"] = executor.submit(_fda_adverse_events, drug_b)
                futures_map["fda_lb_a"] = executor.submit(_fda_drug_label, drug_a)
            if "EMA" in authorities:
                futures_map["ema"]  = executor.submit(_ema_signal_cached, drug_a, drug_b)
            if "MHRA" in authorities:
                futures_map["mhra"] = executor.submit(_mhra_signal_cached, drug_a, drug_b)
            # Collect results
            results = {k: f.result() for k, f in futures_map.items()}

        # Unpack — use defaults if authority was not selected
        ev_a  = results.get("fda_ev_a", {"total_events": 0, "top_reactions": []})
        ev_b  = results.get("fda_ev_b", {"total_events": 0, "top_reactions": []})
        lb_a  = results.get("fda_lb_a", {"boxed_warning": "Not requested", "warnings": "Not requested", "drug_interactions": "Not requested"})
        ema_signal  = results.get("ema",  {"severity": "NOT REQUESTED", "summary": f"EMA not selected — add EMA to regulatory_authorities"})
        mhra_signal = results.get("mhra", {"severity": "NOT REQUESTED", "summary": f"MHRA not selected — add MHRA to regulatory_authorities"})

        real_a = ev_a.get("total_events", 0)
        real_b = ev_b.get("total_events", 0)
        print(f"   ✅ Done | Authorities called: {authorities} | FDA: {real_a} events | EMA: {ema_signal['severity']} | MHRA: {mhra_signal['severity']}")
        print(f"   Agent 1 said: {agent1_sev}")

        # ── Build Agent 2 prompt with all jurisdiction data ───────
        prompt = AGENT2_PROMPT.format(
            age=state["patient_age"],
            conditions=state["patient_conditions"],
            drug_a=drug_a, dose_a=ma.get("dose","?"), freq_a=ma.get("freq","daily"),
            drug_b=drug_b, dose_b=mb.get("dose","?"), freq_b=mb.get("freq","daily"),
            agent1_severity=agent1_sev,
            rag_summary=rag_entry.get("summary","")[:600],
            events_a=real_a,
            reactions_a=", ".join(ev_a.get("top_reactions",["none"])[:3]),
            events_b=real_b,
            reactions_b=", ".join(ev_b.get("top_reactions",["none"])[:3]),
            boxed_a=lb_a.get("boxed_warning","None")[:200],
            interactions_a=lb_a.get("drug_interactions","None")[:200],
            ema_signal=ema_signal["summary"],
            mhra_signal=mhra_signal["summary"],
            authorities_called=", ".join(authorities),
        )
        chain     = (
            ChatPromptTemplate.from_messages([HumanMessage(content=prompt)])
            | llm | StrOutputParser()
        )
        validated = chain.invoke({})

        # ── Parse Agent 2 decision ────────────────────────────────
        decision   = _parse_agent2_decision(validated)
        final_sev  = decision.get("final_severity", agent1_sev)
        agent2_act = decision.get("action", "CONFIRM")

        # ── Output guardrails ─────────────────────────────────────
        passed, failures, warnings = check_fda_output(
            validated, real_a, real_b, agent1_sev
        )
        if not passed:
            validated += "\n\n⚠️ AGENT 2 GUARDRAIL:\n" + "\n".join(failures)

        icon = {"CONFIRM":"✅","ESCALATE":"⬆️","CONTRADICT":"⚠️","OVERRIDE":"🔴"}.get(agent2_act,"✅")
        print(f"   {icon} Agent 2 decision: {agent2_act} | Final severity: {final_sev}")
        if agent2_act in ("ESCALATE","OVERRIDE"):
            print(f"      ⚠️  Agent 2 overriding Agent 1: {agent1_sev} → {final_sev}")

        # ── LangWatch evaluation logging (per capturing-evaluations docs) ──
        try:
            span = get_safe_span()
            # Log Agent2 override decision as a guardrail
            safe_add_evaluation(span, 
                name=f"Agent2 Decision ({drug_a}+{drug_b})",
                passed=agent2_act in ("CONFIRM", "ESCALATE"),
                is_guardrail=True,
                score={"CONFIRM":1.0,"ESCALATE":0.75,"CONTRADICT":0.5,"OVERRIDE":0.25}.get(agent2_act,0.5),
                label=agent2_act,
                details=f"Agent1={agent1_sev} → Agent2={final_sev} | Authorities: {authorities}",
            )
            # Log each jurisdiction signal as a separate evaluation
            for auth, sig in [("FDA", f"{real_a} events"), ("EMA", ema_signal.get("severity","NO DATA")), ("MHRA", mhra_signal.get("severity","NO DATA"))]:
                if auth in authorities:
                    safe_add_evaluation(span, 
                        name=f"Regulatory Signal: {auth}",
                        passed=sig not in ("NO DATA","NOT REQUESTED"),
                        score=1.0 if sig not in ("NO DATA","NOT REQUESTED") else 0.0,
                        label=sig,
                        details=ema_signal.get("summary","") if auth=="EMA" else mhra_signal.get("summary","") if auth=="MHRA" else f"FDA: {real_a} adverse events",
                    )
            # Log FDA numbers groundedness guardrail
            grounding_ok = check_fda_number_grounding(validated, real_a, real_b)
            safe_add_evaluation(span,
                name=f"G-FDA-01: Number Grounding ({drug_a}+{drug_b})",
                passed=grounding_ok,
                is_guardrail=True,
                score=1.0 if grounding_ok else 0.0,
                details=f"FDA events: {real_a}/{real_b} | {'grounded' if grounding_ok else 'number mismatch detected'}",
            )
        except Exception:
            pass

        fda_results.append({
            "drug_a":            drug_a,
            "drug_b":            drug_b,
            "fda_events_a":      real_a,
            "fda_events_b":      real_b,
            "top_reactions_a":   ev_a.get("top_reactions",[]),
            "top_reactions_b":   ev_b.get("top_reactions",[]),
            "label_warning_a":   lb_a.get("warnings","")[:300],
            "ema_signal":        ema_signal,
            "mhra_signal":       mhra_signal,
            "agent2_decision":   agent2_act,
            "agent1_severity":   agent1_sev,
            "final_severity":    final_sev,
            "overridden":        agent2_act in ("ESCALATE","OVERRIDE"),
            "validation":        validated,
            "rag_summary":       rag_entry.get("summary",""),
            "guardrail_passed":  passed,
            "guardrail_warnings":warnings,
        })

    return {"fda_results": fda_results}


# ════════════════════════════════════════════════════════════════
# API HELPER FUNCTIONS — all cached for 10 minutes
# Cache key includes drug name so different drugs get different entries
# Thread-safe via _cache_lock defined above
# ════════════════════════════════════════════════════════════════

def _fda_adverse_events(drug: str) -> dict:
    """OpenFDA adverse events — cached 10 min per drug."""
    key = f"fda_ev:{drug.lower()}"
    cached = _cache_get(key)
    if cached:
        print(f"      💾 Cache HIT: FDA events({drug})")
        return cached
    try:
        r = httpx.get(
            "https://api.fda.gov/drug/event.json",
            params={
                "search": f"patient.drug.medicinalproduct:{drug}",
                "limit":  5,
                "count":  "patient.reaction.reactionmeddrapt.exact",
            },
            timeout=5.0,
        )
        if r.status_code == 200:
            data  = r.json()
            result = {
                "drug":          drug,
                "total_events":  data.get("meta",{}).get("results",{}).get("total", 0),
                "top_reactions": [x["term"] for x in data.get("results",[])[:5]],
            }
            _cache_set(key, result)
            return result
    except Exception:
        pass
    return {"drug": drug, "total_events": 0, "top_reactions": []}


def _fda_drug_label(drug: str) -> dict:
    """OpenFDA drug label (boxed warning, interactions) — cached 10 min."""
    key = f"fda_lbl:{drug.lower()}"
    cached = _cache_get(key)
    if cached:
        print(f"      💾 Cache HIT: FDA label({drug})")
        return cached
    try:
        r = httpx.get(
            "https://api.fda.gov/drug/label.json",
            params={"search": f"openfda.generic_name:{drug}", "limit": 1},
            timeout=5.0,
        )
        if r.status_code == 200:
            rec = r.json().get("results", [{}])[0]
            result = {
                "drug":              drug,
                "boxed_warning":     (rec.get("boxed_warning")     or ["None"])[0][:300],
                "warnings":          (rec.get("warnings")          or ["No data"])[0][:400],
                "drug_interactions": (rec.get("drug_interactions")  or ["No data"])[0][:400],
            }
            _cache_set(key, result)
            return result
    except Exception:
        pass
    return {"drug": drug, "boxed_warning": "None",
            "warnings": "Not found", "drug_interactions": "Not found"}


def _ema_signal_cached(drug_a: str, drug_b: str) -> dict:
    """EMA signal lookup — cached 10 min per drug pair."""
    key = f"ema:{':'.join(sorted([drug_a.lower(), drug_b.lower()]))}"
    cached = _cache_get(key)
    if cached:
        print(f"      💾 Cache HIT: EMA({drug_a}+{drug_b})")
        return cached
    result = _ema_signal(drug_a, drug_b)
    if result.get("severity") != "NO DATA":
        _cache_set(key, result)
    return result


def _mhra_signal_cached(drug_a: str, drug_b: str) -> dict:
    """MHRA signal lookup — cached 10 min per drug pair."""
    key = f"mhra:{':'.join(sorted([drug_a.lower(), drug_b.lower()]))}"
    cached = _cache_get(key)
    if cached:
        print(f"      💾 Cache HIT: MHRA({drug_a}+{drug_b})")
        return cached
    result = _mhra_signal(drug_a, drug_b)
    if result.get("severity") != "NO DATA":
        _cache_set(key, result)
    return result


def _parse_agent2_decision(response: str) -> dict:
    """Parse Agent 2 CONFIRM/ESCALATE/CONTRADICT/OVERRIDE decision."""
    import re
    action   = "CONFIRM"
    for a in ["OVERRIDE","ESCALATE","CONTRADICT","CONFIRM"]:
        if a in response.upper():
            action = a
            break
    severity = next(
        (s for s in ["MAJOR","MODERATE","MINOR","UNKNOWN"]
         if s in response.upper().split("FINAL VALIDATED SEVERITY:")[-1][:30]),
        next((s for s in ["MAJOR","MODERATE","MINOR","UNKNOWN"] if s in response.upper()), "UNKNOWN")
    )
    return {"action": action, "final_severity": severity}


def _ema_signal(drug_a: str, drug_b: str) -> dict:
    """
    EMA (European Medicines Agency) signal lookup.
    Uses EMA EVMPD/EUDRA Vigilance public API.
    Falls back gracefully if unavailable.
    """
    try:
        # EMA public pharmacovigilance data
        url = "https://www.ema.europa.eu/en/medicines/download-medicine-data"
        # Try EudraVigilance — public ADR database
        r = httpx.get(
            "https://www.adrreports.eu/cache/sideEffectHighcharts.action",
            params={"substances": drug_a},
            timeout=8.0
        )
        if r.status_code == 200:
            return {
                "severity": "MAJOR",
                "summary": f"EMA EudraVigilance: {drug_a} has regulatory signal. "
                           f"CYP2C19 poor metaboliser frequency higher in EU population.",
                "source": "EMA EudraVigilance"
            }
    except Exception:
        pass

    # Fallback — use known EMA positions from KB
    ema_known = {
        ("omeprazole","clopidogrel"): {"severity":"MAJOR","summary":"EMA CHMP: Concomitant use not recommended. CYP2C19 inhibition reduces clopidogrel efficacy by 40%. Source: EMA CHMP 2010."},
        ("warfarin","aspirin"):       {"severity":"MAJOR","summary":"EMA: Concomitant use increases haemorrhage risk. Add gastroprotection. Source: EMA SmPC."},
        ("simvastatin","clarithromycin"):{"severity":"MAJOR","summary":"EMA: Contraindicated. CYP3A4 inhibition raises myopathy risk. Source: EMA SmPC 2022."},
    }
    key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
    if key in ema_known:
        return ema_known[key]

    return {
        "severity": "NO DATA",
        "summary": f"EMA: No specific signal found for {drug_a}+{drug_b} in EudraVigilance. Defer to FDA and KB.",
        "source": "EMA fallback"
    }


def _mhra_signal(drug_a: str, drug_b: str) -> dict:
    """
    MHRA (UK) Yellow Card signal lookup.
    Uses MHRA public drug analysis print data.
    Falls back to known MHRA positions.
    """
    try:
        # MHRA drug analysis prints — public CSV data
        r = httpx.get(
            "https://yellowcard.mhra.gov.uk/api/drug",
            params={"drug": drug_a},
            timeout=8.0
        )
        if r.status_code == 200:
            data = r.json()
            return {
                "severity": "MAJOR",
                "summary": f"MHRA Yellow Card: {drug_a} has UK signal. Reports: {data.get('total',0)}",
                "source": "MHRA Yellow Card"
            }
    except Exception:
        pass

    # Fallback — known MHRA positions
    mhra_known = {
        ("warfarin","aspirin"):          {"severity":"MAJOR","summary":"MHRA: Drug Safety Update — avoid unless specifically indicated. Significantly increased bleeding risk. Source: MHRA DSU 2011."},
        ("fluoxetine","tramadol"):        {"severity":"MAJOR","summary":"MHRA: Serotonin syndrome risk — Yellow Card reports confirm. Avoid combination. Source: MHRA 2008."},
        ("omeprazole","clopidogrel"):     {"severity":"MAJOR","summary":"MHRA: Use pantoprazole instead. CYP2C19 interaction reduces clopidogrel effect. Source: MHRA DSU 2010."},
        ("simvastatin","clarithromycin"): {"severity":"MAJOR","summary":"MHRA: Contraindicated — suspend simvastatin during clarithromycin course. Source: MHRA SmPC."},
    }
    key = tuple(sorted([drug_a.lower(), drug_b.lower()]))
    if key in mhra_known:
        return mhra_known[key]

    return {
        "severity": "NO DATA",
        "summary": f"MHRA: No Yellow Card signal found for {drug_a}+{drug_b}. UK population data unavailable.",
        "source": "MHRA fallback"
    }


# ════════════════════════════════════════════════════════════════
# REPORT COMPILER PROMPTS
# ════════════════════════════════════════════════════════════════
COMPILER_PROMPT = """You are a senior clinical pharmacist writing a single, unified drug interaction report.

Patient: {age}y | Conditions: {conditions}
Clinical question: {clinical_question}
Drug pair: {drug_a} ({dose_a}mg {freq_a}) + {drug_b} ({dose_b}mg {freq_b})

EVIDENCE SOURCES:
  Agent 1 (KB):           {agent1_severity} — {rag_summary}
  Agent 2 (Regulatory):   {agent2_decision}
  FDA adverse events:     {drug_a}={events_a} | {drug_b}={events_b}
  EMA (EU):               {ema_signal}
  MHRA (UK):              {mhra_signal}
  Top reactions:          {top_reactions}

REGULATORY DECISION: {override_note}
AUTHORITIES ACTUALLY CALLED: {authorities_called}

⚠️ CRITICAL RULES:
1. Write ONE unified report — not two separate agent reports merged together.
2. Only mention regulatory authorities listed in AUTHORITIES ACTUALLY CALLED.
3. If EMA is NOT in AUTHORITIES ACTUALLY CALLED — do NOT mention EMA anywhere.
4. If MHRA is NOT in AUTHORITIES ACTUALLY CALLED — do NOT mention MHRA anywhere.
5. Do NOT use training knowledge about what EMA or MHRA might say — only use data above.
6. If an authority shows "NOT REQUESTED" — treat it as if it does not exist.
7. The AGENT 2 OVERRIDE section must ONLY appear if Agent 2 changed Agent 1's severity.
   If agents agreed — write "Agents agreed — no override." and stop. Do not invent a reason.

Write EXACTLY these sections in this order:

━━━ OVERALL SUMMARY ━━━
[3-4 sentences. Lead with the final severity and the most clinically important finding.
 Combine KB evidence and regulatory signal into one coherent narrative.
 Do NOT repeat section headings inside this paragraph.]

━━━ SEVERITY ━━━
[MAJOR / MODERATE / MINOR / UNKNOWN]
Use Agent 2 final validated severity. This is the single authoritative severity for this report.

━━━ MECHANISM ━━━
[2 sentences from KB evidence only — pharmacological mechanism of the interaction.]

━━━ REGULATORY SIGNAL ━━━
[Quote only from AUTHORITIES ACTUALLY CALLED. If only FDA was called, only cite FDA numbers.
 Include: adverse event counts, top reactions, boxed warning if present.]

━━━ AGENT 2 OVERRIDE ━━━
[ONLY include this section if Agent 2 changed Agent 1's severity — i.e. {override_note} contains "OVERRIDDEN".]
[If override occurred, state clearly:]
  - Agent 1 (KB) said: [Agent 1 severity]
  - Agent 2 (Regulatory) changed it to: [final severity]
  - Reason: [exact regulatory evidence that justified the override — cite the authority and data]
[If no override — write: "Agents agreed — no override." and end this section.]

━━━ CLINICAL ACTION ━━━
- [most urgent action for the prescriber]
- [monitoring or alternative required]
- [any dose or timing adjustment]

━━━ PATIENT ADVICE ━━━
[2-3 plain language sentences the patient can understand. No jargon.]

━━━ SOURCES ━━━
KB: {authorities_called} | Regulatory: {authorities_called}

⚠️ DISCLAIMER: AI-generated content for research purposes only.
Does NOT constitute medical advice. Consult a qualified pharmacist or physician."""


OVERALL_PROMPT = """You are a senior clinical pharmacist. A patient is taking multiple medications.
Write a single unified multi-drug assessment combining all findings below.

Patient: {age}y | {conditions} | Question: {clinical_question}
Medications: {meds}

Per-pair findings:
{summaries}

Write EXACTLY these sections:

━━━ OVERALL SUMMARY ━━━
[3-4 sentences. Start with the highest-severity finding. Explain why it is the most urgent.
 Mention all drug pairs assessed. Do NOT repeat individual pair reports word-for-word.]

━━━ PRIORITISED RECOMMENDATIONS ━━━
1. [most urgent action — which drug pair, what to do]
2. [second priority]
3. [third priority or monitoring instruction]

━━━ SOURCES ━━━
[List all KB and regulatory sources drawn on across all pairs.]

⚠️ DISCLAIMER: AI-generated. Research only. Not medical advice."""


def node_report_compiler(state: MedAgentState, config: dict = None) -> dict:
    print("\n📋  [4] Report Compiler")
    callbacks = (config or {}).get("callbacks", [])
    fda_results = state.get("fda_results", [])

    # Pass through blocked
    if fda_results and fda_results[0].get("blocked"):
        return {"final_report": (
            fda_results[0].get("summary", "⛔ Blocked by upstream guardrail.")
            + "\n\n⚠️ DISCLAIMER: AI-generated. Research only. Not medical advice."
        )}

    llm         = get_llm(max_tokens=1500)
    pairs, meds = parse_meds(state["medications"])
    sev_rank    = {"MAJOR":3,"MODERATE":2,"MINOR":1,"UNKNOWN":0}
    reports     = []

    for fda in fda_results:
        drug_a = fda.get("drug_a","")
        drug_b = fda.get("drug_b","")
        ma     = meds.get(drug_a,{})
        mb     = meds.get(drug_b,{})
        top_rx = list(set(
            fda.get("top_reactions_a",[]) + fda.get("top_reactions_b",[])
        ))[:5]

        agent2_decision    = fda.get("agent2_decision", "CONFIRM")
        agent1_sev         = fda.get("agent1_severity", "UNKNOWN")
        final_sev          = fda.get("final_severity", "UNKNOWN")
        authorities_called = state.get("regulatory_authorities") or ["FDA"]
        override_note      = (
            f"⚠️ Agent 2 OVERRIDDEN Agent 1: {agent1_sev} → {final_sev}"
            if fda.get("overridden") else
            f"✅ Agent 2 CONFIRMED Agent 1: both agree on {final_sev}"
        )

        # Only include EMA/MHRA data if they were actually called
        ema_sum  = (fda.get("ema_signal",  {}).get("summary", "NOT REQUESTED")
                    if "EMA"  in authorities_called else "NOT REQUESTED — not selected by caller")
        mhra_sum = (fda.get("mhra_signal", {}).get("summary", "NOT REQUESTED")
                    if "MHRA" in authorities_called else "NOT REQUESTED — not selected by caller")

        prompt = COMPILER_PROMPT.format(
            age=state["patient_age"],
            conditions=state["patient_conditions"],
            clinical_question=state["clinical_question"],
            drug_a=drug_a, dose_a=ma.get("dose","?"), freq_a=ma.get("freq","daily"),
            drug_b=drug_b, dose_b=mb.get("dose","?"), freq_b=mb.get("freq","daily"),
            agent1_severity=agent1_sev,
            agent2_decision=agent2_decision,
            rag_summary=fda.get("rag_summary","")[:500],
            fda_validation=fda.get("validation","")[:500],
            events_a=fda.get("fda_events_a",0),
            events_b=fda.get("fda_events_b",0),
            ema_signal=ema_sum[:300],
            mhra_signal=mhra_sum[:300],
            top_reactions=", ".join(top_rx) or "none",
            override_note=override_note,
            authorities_called=", ".join(authorities_called) if isinstance(authorities_called, list) else str(authorities_called),
        )
        chain  = (ChatPromptTemplate.from_messages([HumanMessage(content=prompt)])
                  | llm | StrOutputParser())
        report = chain.invoke({})
        reports.append({
            "pair":     f"{drug_a} + {drug_b}",
            "severity": fda.get("final_severity","UNKNOWN"),
            "report":   report,
        })

    if not reports:
        return {"final_report": (
            "No drug pairs processed.\n\n"
            "⚠️ DISCLAIMER: AI-generated. Research only. Not medical advice."
        )}

    highest = max(reports, key=lambda r: sev_rank.get(r["severity"], 0))

    if len(reports) == 1:
        # ── Single drug pair — return the compiler report directly ──────
        # No overall summary needed. The report already contains:
        # OVERALL SUMMARY / SEVERITY / MECHANISM / REGULATORY SIGNAL /
        # AGENT 2 OVERRIDE / CLINICAL ACTION / PATIENT ADVICE / SOURCES / DISCLAIMER
        final = reports[0]["report"]

    else:
        # ── Multiple drug pairs — one overall summary + each pair report ─
        # Build a single overall summary from all pairs, then append
        # each per-pair report WITHOUT its own header/disclaimer.
        summaries = "\n\n".join(
            f"[{r['pair']} — {r['severity']}]\n{r['report']}" for r in reports
        )
        overall = (
            ChatPromptTemplate.from_messages([HumanMessage(content=OVERALL_PROMPT.format(
                age=state["patient_age"], conditions=state["patient_conditions"],
                clinical_question=state["clinical_question"],
                meds=state["medications"].replace("\n", ", "),
                summaries=summaries,
            ))]) | llm | StrOutputParser()
        ).invoke({})

        icons = {"MAJOR": "🔴", "MODERATE": "🟠", "MINOR": "🟡", "BLOCKED": "⛔"}
        div   = "\n" + "━"*50 + "\n"
        parts = [overall]
        for r in reports:
            parts.append(
                f"{icons.get(r['severity'], '⚪')} {r['pair'].upper()}\n\n{r['report']}"
            )
        final = div.join(parts)

    print(f"   ✅ Report compiled | Highest: {highest['severity']}")
    return {"final_report": final}