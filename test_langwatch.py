"""
LangWatch test — uses direct OpenAI (not LangChain) to isolate the issue.
Usage: python test_langwatch.py
"""
import os, sys, time
 
# ── Paste ONLY your OpenAI key here — LangWatch key is already set ────
OPENAI_KEY    = "sk-proj"
LANGWATCH_KEY = "sk-lw-"

# Must set BEFORE importing langwatch
os.environ["OPENAI_API_KEY"]    = OPENAI_KEY
os.environ["LANGWATCH_API_KEY"] = LANGWATCH_KEY
 
print("=" * 55)
print("STEP 1 — LangWatch setup")
print("=" * 55)
 
import langwatch
langwatch.setup()
print(f"✅ Setup done")
 
print("\n" + "=" * 55)
print("STEP 2 — Sending trace using direct OpenAI client")
print("(bypasses LangChain to isolate any issues)")
print("=" * 55)
 
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
 
    @langwatch.trace(name="MedAgentPipeline")
    def run_pipeline():
        # Tell LangWatch to auto-track all OpenAI calls
        langwatch.get_current_trace().autotrack_openai_calls(client)
 
        # Simulate Agent 1
        with langwatch.span(name="Agent1_RAGReader", type="tool") as span1:
            resp1 = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"user","content":"Warfarin + Aspirin interaction in one sentence?"}],
                max_tokens=30,
            )
            result1 = resp1.choices[0].message.content
            span1.update(output=result1)
 
        # Simulate Agent 2
        with langwatch.span(name="Agent2_FDAValidator", type="tool") as span2:
            resp2 = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"user","content":"FDA signal for warfarin in one sentence?"}],
                max_tokens=30,
            )
            result2 = resp2.choices[0].message.content
            span2.update(output=result2)
 
        # Simulate Report Compiler
        with langwatch.span(name="ReportCompiler", type="tool") as span3:
            final = f"SEVERITY: MAJOR\n{result1}\n{result2}"
            span3.update(output=final)
 
        return final
 
    output = run_pipeline()
    print(f"✅ Pipeline ran: {output[:60]}...")
 
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
 
print("\n" + "=" * 55)
print("STEP 3 — Flushing traces to LangWatch...")
print("=" * 55)
 
# Wait for traces to be sent
print("✅ Traces sent")
 
time.sleep(3)
print("✅ Done — waiting 3 seconds for LangWatch to process")
 
print("\n" + "=" * 55)
print("Now go to: https://app.langwatch.ai")
print("Click 'Traces' in left menu")
print("You should see 'MedAgentPipeline' with 3 spans inside")
print("=" * 55)
 
# Write .env
env = f"""OPENAI_API_KEY={OPENAI_KEY}
LANGWATCH_API_KEY={LANGWATCH_KEY}
APP_PORT=8000
KB_DIR=./medagent_kb_docs
"""
open('.env','w').write(env)
print("✅ .env written")
