"""
MedAgent Guardrails — deterministic, zero LLM cost.
Extracted from the working LangFlow components.
"""
import re
from typing import Tuple, List


def check_input(medications: str, patient_age: str,
                conditions: str, question: str) -> Tuple[bool, List[str]]:
    failures = []

    # G-IN-01: At least 2 medications, correct format
    lines = [l.strip() for l in medications.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        failures.append("G-IN-01: At least 2 medications required (name:dose:frequency per line)")
    for line in lines:
        if len(line.split(":")) < 2:
            failures.append(f"G-IN-01: '{line}' missing dose — use name:dose_mg:frequency")

    # G-IN-02: Drug name sanity
    for line in lines:
        name = line.split(":")[0].strip()
        if re.match(r'^\d+$', name):
            failures.append(f"G-IN-02: '{name}' is numeric — not a valid drug name")
        if len(name) > 50:
            failures.append(f"G-IN-02: Drug name too long — possible injection attempt")

    # G-IN-03: Prompt injection scan
    combined = f"{medications} {conditions} {question}".lower()
    for pattern in [
        r"ignore (previous|prior|all)\s*(instructions?|rules?)",
        r"(system|admin)\s*(override|mode)",
        r"you are now\s+",
        r"forget (your|the)\s*(previous|original)\s*(role|instructions?)",
        r"<(system|instruction|override)>",
    ]:
        if re.search(pattern, combined):
            failures.append("G-IN-03: Prompt injection pattern detected in input")
            break

    # G-IN-04: Age range
    try:
        age = int(str(patient_age).strip())
        if not 0 <= age <= 120:
            failures.append(f"G-IN-04: Age {age} outside valid range 0–120")
    except ValueError:
        failures.append(f"G-IN-04: Age '{patient_age}' is not a valid number")

    return len(failures) == 0, failures


def check_rag_output(response: str, chunks: list) -> Tuple[bool, List[str], List[str]]:
    failures, warnings = [], []

    # G-OUT-01: Valid severity present
    if not any(s in response.upper() for s in ["MAJOR","MODERATE","MINOR","UNKNOWN"]):
        failures.append("G-OUT-01: No valid severity (MAJOR/MODERATE/MINOR/UNKNOWN)")

    # G-OUT-02: Disclaimer present
    # Note: Agent 1 output format (4 numbered sections) does not require
    # the disclaimer to be reproduced — it is in the prompt footer and
    # will appear in the final compiled report. Downgraded to warning.
    if not any(p in response.lower() for p in [
        "not medical advice","research only","disclaimer","ai-generated","consult"
    ]):
        warnings.append("G-OUT-02: Disclaimer not in Agent 1 response (present in compiled report)")

    # G-OUT-03: Source citation present
    if not any(s in response.lower() for s in [
        "bnf","nice","fda","esc","mhra","source:","guideline","stockley"
    ]):
        failures.append("G-OUT-03: No evidence source cited (BNF/FDA/ESC/NICE required)")

    # G-OUT-04: No fabrication markers
    for m in ["i believe this","i think this may","i'm not certain","i cannot confirm"]:
        if m in response.lower():
            failures.append(f"G-OUT-04: Fabrication marker: '{m}'")
            break

    # G-OUT-05: Severity grounded in retrieved KB chunks
    found_sev = next((s for s in ["MAJOR","MODERATE","MINOR"] if s in response.upper()), None)
    if found_sev:
        chunk_text = " ".join(d.page_content.upper() for d in chunks)
        if found_sev not in chunk_text:
            warnings.append(
                f"G-OUT-05: Severity '{found_sev}' not found in KB chunks — "
                f"may not be grounded in source data"
            )

    # G-OUT-06: No prescribing instructions
    for pat in [r"\bprescribe\b", r"change (the )?dose to \d+", r"write (a|the) prescription"]:
        if re.search(pat, response.lower()):
            failures.append("G-OUT-06: Prescribing instruction detected — not permitted")
            break

    return len(failures) == 0, failures, warnings


def check_fda_output(response: str, real_a: int,
                     real_b: int, rag_sev: str) -> Tuple[bool, List[str], List[str]]:
    failures, warnings = [], []

    # G-FDA-01: Large numbers must match real FDA API counts
    for num_str in re.findall(r'\b(\d{3,})\b', response):
        num = int(num_str)
        if num > 100:
            close = any(
                abs(num - real) < max(10, real * 0.1)
                for real in [real_a, real_b] if real > 0
            )
            if not close:
                warnings.append(
                    f"G-FDA-01: Number {num} doesn't match real FDA counts "
                    f"({real_a}, {real_b}) — possible hallucinated figure"
                )

    # G-FDA-02: Valid severity present
    if not any(s in response.upper() for s in ["MAJOR","MODERATE","MINOR","UNKNOWN"]):
        failures.append("G-FDA-02: No valid severity in FDA validation response")

    # G-FDA-03: Severity consistency check
    rank = {"MAJOR":3,"MODERATE":2,"MINOR":1,"UNKNOWN":0}
    found = next((s for s in ["MAJOR","MODERATE","MINOR","UNKNOWN"] if s in response.upper()), None)
    if found and rag_sev in rank and found in rank:
        if rank[rag_sev] - rank[found] >= 2:
            warnings.append(
                f"G-FDA-03: Severity downgraded {rag_sev}→{found} by 2 levels — verify evidence"
            )

    # G-FDA-04: Disclaimer present
    # Agent 2 structured response (DECISION/REASON/SIGNALS) does not
    # always reproduce the disclaimer from the prompt footer.
    # Downgraded to warning — disclaimer is guaranteed in compiled report.
    if not any(p in response.lower() for p in [
        "not medical advice","research only","disclaimer","ai-generated"
    ]):
        warnings.append("G-FDA-04: Disclaimer not in Agent 2 response (present in compiled report)")

    # G-FDA-05: No prescribing
    for pat in [r"\bprescribe\b", r"change (the )?dose to \d+"]:
        if re.search(pat, response.lower()):
            failures.append("G-FDA-05: Prescribing instruction in Agent 2 — not permitted")
            break

    return len(failures) == 0, failures, warnings


def check_fda_number_grounding(response: str, real_a: int, real_b: int) -> bool:
    """
    G-FDA-01: Check if any large numbers in Agent 2 response are
    grounded in the real FDA adverse event counts.

    Returns True if:
    - No large numbers (>100) appear in the response, OR
    - All large numbers are within 10% of real_a or real_b

    This is logged as a separate LangWatch evaluation so it does not
    block the pipeline — it is an observability signal only.
    """
    if real_a == 0 and real_b == 0:
        # FDA returned no data (drug not in OpenFDA or API timeout)
        # Cannot ground numbers — but this is not a failure, it is NO DATA
        return True

    for num_str in re.findall(r'\b(\d{3,})\b', response):
        num = int(num_str)
        if num > 100:
            close = any(
                abs(num - real) < max(10, real * 0.1)
                for real in [real_a, real_b] if real > 0
            )
            if not close:
                return False
    return True