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
from observability import traced_node
from state import MedAgentState
from guardrails import check_input, check_rag_output, check_fda_output

load_dotenv(override=False)

# ── LangWatch setup ───────────────────────────────────────────────────
# Get your free API key at https://app.langwatch.ai
# Set LANGWATCH_API_KEY in your .env file
# LangWatch setup happens in api.py — not here
# This prevents the "already setup" conflict and 401 errors
_lw_key = os.getenv("LANGWATCH_API_KEY", "")

# ── KB docs — same as in the working LangFlow components ─────────────
KB_DOCS = [
    {"text": "Warfarin + aspirin: MAJOR interaction. Warfarin inhibits Vitamin K clotting factors (II,VII,IX,X). Aspirin irreversibly inhibits COX-1, reducing thromboxane A2-mediated platelet aggregation. Additive effect dramatically raises GI and intracranial bleeding risk. If unavoidable: add PPI, increase INR monitoring frequency. Source: BNF + Clinical Pharmacology.", "meta": {"drugs": "warfarin,aspirin", "severity": "MAJOR"}},
    {"text": "Warfarin + ibuprofen: MAJOR. NSAIDs inhibit COX-1/COX-2, reduce gastric mucosal protection, may displace warfarin from protein binding. Avoid; use paracetamol instead. Source: BNF.", "meta": {"drugs": "warfarin,ibuprofen", "severity": "MAJOR"}},
    {"text": "Omeprazole + clopidogrel: MAJOR. Omeprazole inhibits CYP2C19, reducing clopidogrel activation by up to 47%. Risk of stent thrombosis in post-ACS patients. Switch to pantoprazole. Source: ESC Guidelines.", "meta": {"drugs": "omeprazole,clopidogrel", "severity": "MAJOR"}},
    {"text": "Fluoxetine + tramadol: MAJOR — serotonin syndrome risk. Fluoxetine inhibits SERT and CYP2D6. Avoid combination. Source: FDA MedWatch.", "meta": {"drugs": "fluoxetine,tramadol", "severity": "MAJOR"}},
    {"text": "Metformin + alcohol: MODERATE. Heavy alcohol with metformin increases lactic acidosis risk. Source: MHRA.", "meta": {"drugs": "metformin,alcohol", "severity": "MODERATE"}},
    {"text": "Severity definitions — MAJOR: avoid combination. MODERATE: use with caution. MINOR: clinically minimal.", "meta": {"drugs": "general", "severity": "INFO"}},
]

# ── LLM singleton ─────────────────────────────────────────────────────
def get_llm(max_tokens=1200, callbacks=None):
    load_dotenv(override=False)  # never override shell $env: vars
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not found. Set it in .env")
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        api_key=key,
        max_tokens=max_tokens,
        callbacks=callbacks or [],
    )

# ── ChromaDB singleton ────────────────────────────────────────────────
_db = None

def get_chromadb():
    global _db
    if _db:
        return _db
    kb_dir    = os.getenv("KB_DIR", "./medagent_kb_docs")
    txt_files = glob.glob(f"{kb_dir}/*.txt")
    raw_docs  = []
    for fpath in txt_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                raw_docs.append(Document(
                    page_content=f.read(),
                    metadata={"source": os.path.basename(fpath)}
                ))
        except Exception as e:
            print(f"KB load error {fpath}: {e}")
    if not raw_docs:
        # Fall back to inline KB — same as working LangFlow component
        raw_docs = [Document(page_content=d["text"], metadata=d["meta"]) for d in KB_DOCS]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(raw_docs)
    emb = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY", "")
    )
    _db = Chroma.from_documents(chunks, emb,
        collection_name="medagent_openai",
        persist_directory="./chroma_medagent_openai")
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

Respond with:
1. SEVERITY: [MAJOR / MODERATE / MINOR / UNKNOWN]
2. SUMMARY: 2 sentences drawn from context
3. CLINICAL ACTION: 2-3 bullet points
4. SOURCE: quote the source tag from context (e.g. BNF 2024)

⚠️ DISCLAIMER: AI-generated. Research only. Not medical advice."""

@traced_node("Agent1_RAGReader")
def node_agent1_rag(state: MedAgentState, config: dict = None) -> dict:
    print("\n🔍  [2] Agent 1 — RAG Reader")
    callbacks = (config or {}).get("callbacks", [])

    # RxNorm validation — same logic as working LangFlow component
    pairs, meds = parse_meds(state["medications"])
    blocked = []
    validation_log = []

    for drug in list(meds.keys()):
        result = _rxnorm_validate(drug)
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
                f"⛔ BLOCKED by RxNorm MCP Tool.\n"
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

    

    return {"rag_results": results}


def _rxnorm_validate(drug_name: str) -> dict:
    """Same RxNorm logic as in working LangFlow Agent 1 component."""
    try:
        r = httpx.get(
            "https://rxnav.nlm.nih.gov/REST/rxcui.json",
            params={"name": drug_name, "search": 2}, timeout=5.0
        )
        if r.status_code == 200:
            rxcui = r.json().get("idGroup", {}).get("rxnormId", [])
            if rxcui:
                return {"drug": drug_name, "found": True,
                        "status": f"✅ RxNorm validated — RxCUI: {rxcui[0]}"}
            return {"drug": drug_name, "found": False,
                    "status": f"❌ NOT in RxNorm — possibly fictional or misspelled"}
    except Exception:
        pass
    return {"drug": drug_name, "found": True,
            "status": "⚠️ RxNorm unreachable — proceeding without validation"}


# ════════════════════════════════════════════════════════════════════
# NODE 3 — AGENT 2: FDA VALIDATOR
# Direct port of MedAgentFDAValidator from working LangFlow JSON
# LangSmith span: "Agent2_FDAValidator"
# ════════════════════════════════════════════════════════════════════
AGENT2_PROMPT = """You are a drug safety specialist cross-checking a drug interaction report.

Patient: {age}y | Conditions: {conditions}
Drugs: {drug_a} ({dose_a}mg {freq_a}) + {drug_b} ({dose_b}mg {freq_b})

AGENT 1 KNOWLEDGE BASE FINDINGS (already verified from KB):
{rag_summary}

LIVE OPENFDA DATA (retrieved now — use these exact numbers):
  {drug_a}: {events_a} adverse event reports | Top reactions: {reactions_a}
  {drug_b}: {events_b} adverse event reports | Top reactions: {reactions_b}
  {drug_a} boxed warning: {boxed_a}
  {drug_a} label warnings: {warnings_a}
  Drug interaction text ({drug_a} label): {interactions_a}

Rules:
- Quote the exact FDA event numbers above — do not invent figures
- If FDA data confirms Agent 1 findings, state this explicitly
- Base FINAL VALIDATED SEVERITY on both KB and FDA data combined

Respond with:
1. VALIDATION: Does FDA data confirm or contradict KB findings?
2. FDA SIGNAL: Quote exact event count. Above 10000 is a strong signal.
3. DOSE-SPECIFIC RISK: Does patient dose affect risk?
4. PATIENT FACTORS: How do age and conditions modulate risk?
5. FINAL VALIDATED SEVERITY: [MAJOR / MODERATE / MINOR / UNKNOWN]

⚠️ DISCLAIMER: AI-generated. Research only. Not medical advice."""

@traced_node("Agent2_FDAValidator")
def node_agent2_fda(state: MedAgentState, config: dict = None) -> dict:
    print("\n🔬  [3] Agent 2 — FDA Validator")
    callbacks = (config or {}).get("callbacks", [])
    rag_results = state.get("rag_results", [])

    # Pass through blocked signals unchanged
    if rag_results and rag_results[0].get("blocked"):
        return {"fda_results": rag_results}

    llm = get_llm(callbacks=callbacks)
    pairs, meds = parse_meds(state["medications"])
    fda_results = []

    for i, (drug_a, drug_b) in enumerate(pairs):
        rag_entry = rag_results[i] if i < len(rag_results) else rag_results[0]
        ma = meds.get(drug_a, {})
        mb = meds.get(drug_b, {})

        # FDA API calls — same sync httpx pattern as working component
        ev_a = _fda_adverse_events(drug_a)
        ev_b = _fda_adverse_events(drug_b)
        lb_a = _fda_drug_label(drug_a)

        real_a = ev_a.get("total_events", 0)
        real_b = ev_b.get("total_events", 0)

        prompt = AGENT2_PROMPT.format(
            age=state["patient_age"],
            conditions=state["patient_conditions"],
            drug_a=drug_a, dose_a=ma.get("dose","?"), freq_a=ma.get("freq","daily"),
            drug_b=drug_b, dose_b=mb.get("dose","?"), freq_b=mb.get("freq","daily"),
            rag_summary=rag_entry.get("summary","")[:800],
            events_a=real_a, reactions_a=", ".join(ev_a.get("top_reactions",["none"])),
            events_b=real_b, reactions_b=", ".join(ev_b.get("top_reactions",["none"])),
            boxed_a=lb_a.get("boxed_warning","None")[:250],
            warnings_a=lb_a.get("warnings","None")[:300],
            interactions_a=lb_a.get("drug_interactions","None")[:300],
        )
        chain     = (ChatPromptTemplate.from_messages([HumanMessage(content=prompt)])
                     | llm | StrOutputParser())
        validated = chain.invoke({})

        # Output guardrails
        passed, failures, warnings = check_fda_output(
            validated, real_a, real_b, rag_entry.get("severity","UNKNOWN")
        )
        if not passed:
            validated += "\n\n⚠️ AGENT 2 GUARDRAIL:\n" + "\n".join(failures)

        final_sev = next(
            (s for s in ["MAJOR","MODERATE","MINOR","UNKNOWN"] if s in validated.upper()),
            rag_entry.get("severity","UNKNOWN")
        )
        fda_results.append({
            "drug_a": drug_a, "drug_b": drug_b,
            "fda_events_a": real_a, "fda_events_b": real_b,
            "top_reactions_a": ev_a.get("top_reactions",[]),
            "top_reactions_b": ev_b.get("top_reactions",[]),
            "label_warning_a": lb_a.get("warnings","")[:300],
            "validation": validated,
            "final_severity": final_sev,
            "rag_summary": rag_entry.get("summary",""),
            "guardrail_passed": passed,
            "guardrail_warnings": warnings,
        })
        icon = "✅" if passed else "⚠️"
        print(f"   {icon} {drug_a}+{drug_b} → {final_sev} | FDA: {real_a}/{real_b} events")

    

    return {"fda_results": fda_results}


def _fda_adverse_events(drug: str) -> dict:
    try:
        r = httpx.get("https://api.fda.gov/drug/event.json",
            params={"search": f"patient.drug.medicinalproduct:{drug}",
                    "limit": 5, "count": "patient.reaction.reactionmeddrapt.exact"},
            timeout=10.0)
        if r.status_code == 200:
            data = r.json()
            return {"drug": drug,
                    "total_events": data.get("meta",{}).get("results",{}).get("total",0),
                    "top_reactions": [x["term"] for x in data.get("results",[])[:5]]}
    except Exception: pass
    return {"drug": drug, "total_events": 0, "top_reactions": []}


def _fda_drug_label(drug: str) -> dict:
    try:
        r = httpx.get("https://api.fda.gov/drug/label.json",
            params={"search": f"openfda.generic_name:{drug}", "limit": 1},
            timeout=10.0)
        if r.status_code == 200:
            rec = r.json().get("results",[{}])[0]
            return {"drug": drug,
                    "boxed_warning": (rec.get("boxed_warning") or ["None"])[0][:300],
                    "warnings":      (rec.get("warnings")      or ["No data"])[0][:400],
                    "drug_interactions": (rec.get("drug_interactions") or ["No data"])[0][:400]}
    except Exception: pass
    return {"drug": drug, "boxed_warning": "None",
            "warnings": "Not found", "drug_interactions": "Not found"}


# ════════════════════════════════════════════════════════════════════
# NODE 4 — REPORT COMPILER
# Direct port of MedAgentReportCompiler from working LangFlow JSON
# LangSmith span: "ReportCompiler"
# ════════════════════════════════════════════════════════════════════
COMPILER_PROMPT = """You are a senior clinical pharmacist writing a formal interaction report.

Patient: {age}y | Conditions: {conditions}
Clinical question: {clinical_question}

Drug pair: {drug_a} ({dose_a}mg {freq_a}) + {drug_b} ({dose_b}mg {freq_b})
Knowledge Base (Agent 1): {rag_summary}
FDA Validation (Agent 2): {fda_validation}
FDA adverse events: {drug_a}={events_a} | {drug_b}={events_b}
Top reactions: {top_reactions}

Write exactly these sections:

SEVERITY: [MAJOR/MODERATE/MINOR/UNKNOWN]

SUMMARY:
[2-3 sentences]

MECHANISM:
[2 sentences]

DOSE CONSIDERATION:
[1 sentence]

CLINICAL ACTION:
- [point 1]
- [point 2]
- [point 3]

PATIENT ADVICE:
[2-3 plain language sentences]

SOURCES: Knowledge base + OpenFDA (open.fda.gov)"""

OVERALL_PROMPT = """Summarise this multi-drug interaction assessment.
Patient: {age}y | {conditions} | Question: {clinical_question}
Medications: {meds}
Findings:
{summaries}

OVERALL SUMMARY:
[3-4 sentences, most clinically important first]

PRIORITISED RECOMMENDATIONS:
1. [most urgent]
2. [second]
3. [third]"""

@traced_node("ReportCompiler")
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

        prompt = COMPILER_PROMPT.format(
            age=state["patient_age"],
            conditions=state["patient_conditions"],
            clinical_question=state["clinical_question"],
            drug_a=drug_a, dose_a=ma.get("dose","?"), freq_a=ma.get("freq","daily"),
            drug_b=drug_b, dose_b=mb.get("dose","?"), freq_b=mb.get("freq","daily"),
            rag_summary=fda.get("rag_summary","")[:800],
            fda_validation=fda.get("validation","")[:800],
            events_a=fda.get("fda_events_a",0),
            events_b=fda.get("fda_events_b",0),
            top_reactions=", ".join(top_rx) or "none",
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

    highest   = max(reports, key=lambda r: sev_rank.get(r["severity"],0))
    summaries = "\n\n".join(
        f"[{r['pair']} — {r['severity']}]\n{r['report'][:400]}" for r in reports
    )
    overall = (
        ChatPromptTemplate.from_messages([HumanMessage(content=OVERALL_PROMPT.format(
            age=state["patient_age"], conditions=state["patient_conditions"],
            clinical_question=state["clinical_question"],
            meds=state["medications"].replace("\n",", "),
            summaries=summaries,
        ))]) | llm | StrOutputParser()
    ).invoke({})

    # Build clean plain text — same format as working LangFlow component
    div   = "\n" + "="*55 + "\n"
    icons = {"MAJOR":"🔴","MODERATE":"🟠","MINOR":"🟡","BLOCKED":"⛔"}
    parts = [
        f"🏥 MEDAGENT AI — DRUG INTERACTION REPORT\n"
        f"Patient: {state['patient_age']}y | {state['patient_conditions']}\n"
        f"Highest severity: {highest['severity']}"
    ]
    for r in reports:
        parts.append(f"{icons.get(r['severity'],'⚪')} {r['pair'].upper()}\n\n{r['report']}")
    parts.append(overall)
    parts.append(
        "⚠️ DISCLAIMER: AI-generated content for research purposes only. "
        "Does NOT constitute medical advice. Consult a qualified pharmacist or physician."
    )
    final = div.join(parts)
    print(f"   ✅ Report compiled | Highest: {highest['severity']}")

   

    return {"final_report": final}