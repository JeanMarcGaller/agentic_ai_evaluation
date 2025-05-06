from dotenv import load_dotenv
load_dotenv()

import json
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from bert_score import score as bert_score  # Optional

# Eingabedaten laden
with open("evaluation_inputs.json", "r") as f:
    data = json.load(f)

question = data["question"]
responder_output = data["responder_answer"]
revisor_output = data["revisor_answer"]
gold_reference = data.get("gold_answer", "").strip()

# LLM für Evaluatoren (für LLM-basierte Evaluationsmethoden)
llm = ChatOpenAI(model="o4-mini")

# Neue Evaluatoren laden (LangChain Smith Evaluator-System)
qa_eval = load_evaluator("qa", llm=llm)
helpfulness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "helpfulness"})

# Exact Match
def exact_match(pred, ref):
    return pred.strip().lower() == ref.strip().lower()

# Optional: BERTScore
def compute_bertscore(pred, ref, lang="en"):
    P, R, F1 = bert_score([pred], [ref], lang=lang)
    return {"precision": round(P[0].item(), 3), "recall": round(R[0].item(), 3), "f1": round(F1[0].item(), 3)}

# Auswertung
if gold_reference:
    print("\n📊 RESPONDER vs. GOLD")
    print("✅ Exact Match:", exact_match(responder_output, gold_reference))
    print("🤖 QAEval:", qa_eval.evaluate_strings(
        input=question,
        prediction=responder_output,
        reference=gold_reference
    ))
    print("🧠 Helpfulness:", helpfulness_eval.evaluate_strings(
        input=question,
        prediction=responder_output,
        reference=gold_reference
    ))
    print("📐 BERTScore:", compute_bertscore(responder_output, gold_reference))

    print("\n📊 REVISOR vs. GOLD")
    print("✅ Exact Match:", exact_match(revisor_output, gold_reference))
    print("🤖 QAEval:", qa_eval.evaluate_strings(
        input=question,
        prediction=revisor_output,
        reference=gold_reference
    ))
    print("🧠 Helpfulness:", helpfulness_eval.evaluate_strings(
        input=question,
        prediction=revisor_output,
        reference=gold_reference
    ))
    print("📐 BERTScore:", compute_bertscore(revisor_output, gold_reference))
else:
    print("⚠️ Kein Goldstandard vorhanden – bitte `gold_answer` in evaluation_inputs.json eintragen.")
