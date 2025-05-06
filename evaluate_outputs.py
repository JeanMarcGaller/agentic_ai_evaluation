# === evaluate_outputs.py ===

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

# LLM für Evaluatoren
llm = ChatOpenAI(model="gpt-4") # This chain was only tested with GPT-4. Performance may be significantly worse with other models.

# LangChain-Evaluatoren laden
cot_qa_eval = load_evaluator("cot_qa", llm=llm)
helpfulness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "helpfulness"})
correctness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "correctness"})
conciseness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "conciseness"})
coherence_eval = load_evaluator("criteria", llm=llm, config={"criteria": "coherence"})
relevance_eval = load_evaluator("criteria", llm=llm, config={"criteria": "relevance"})
pairwise_eval = load_evaluator("pairwise_string", llm=llm, config={"criteria": "overall"})

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
    print("🤖 CotQaEval:", cot_qa_eval.evaluate_strings(
        input=question,
        prediction=responder_output,
        reference=gold_reference
    ))
    print("🧠 Helpfulness:", helpfulness_eval.evaluate_strings(
        input=question,
        prediction=responder_output,
        reference=gold_reference
    ))
    print("🎯 Correctness:", correctness_eval.evaluate_strings(
        input=question,
        prediction=responder_output,
        reference=gold_reference
    ))
    print("✂️ Conciseness:", conciseness_eval.evaluate_strings(
        input=question,
        prediction=responder_output,
        reference=gold_reference
    ))
    print("🔗 Coherence:", coherence_eval.evaluate_strings(
        input=question,
        prediction=responder_output,
        reference=gold_reference
    ))
    print("📌 Relevance:", relevance_eval.evaluate_strings(
        input=question,
        prediction=responder_output,
        reference=gold_reference
    ))
    print("📐 BERTScore:", compute_bertscore(responder_output, gold_reference))

    print("\n📊 REVISOR vs. GOLD")
    print("✅ Exact Match:", exact_match(revisor_output, gold_reference))
    print("🤖 CotQaEval:", cot_qa_eval.evaluate_strings(
        input=question,
        prediction=revisor_output,
        reference=gold_reference
    ))
    print("🧠 Helpfulness:", helpfulness_eval.evaluate_strings(
        input=question,
        prediction=revisor_output,
        reference=gold_reference
    ))
    print("🎯 Correctness:", correctness_eval.evaluate_strings(
        input=question,
        prediction=revisor_output,
        reference=gold_reference
    ))
    print("✂️ Conciseness:", conciseness_eval.evaluate_strings(
        input=question,
        prediction=revisor_output,
        reference=gold_reference
    ))
    print("🔗 Coherence:", coherence_eval.evaluate_strings(
        input=question,
        prediction=revisor_output,
        reference=gold_reference
    ))
    print("📌 Relevance:", relevance_eval.evaluate_strings(
        input=question,
        prediction=revisor_output,
        reference=gold_reference
    ))
    print("📐 BERTScore:", compute_bertscore(revisor_output, gold_reference))

    print("\n⚖️ REVISOR vs. RESPONDER (Pairwise)")
    pairwise_result = pairwise_eval.evaluate_string_pairs(
        input=question,
        prediction=responder_output,
        prediction_b=revisor_output
    )
    print("🔍 Overall Comparison:", pairwise_result)

    winner = pairwise_result.get("value", "").lower()
    if winner == "a":
        print("✅ Gewinner: Responder")
    elif winner == "b":
        print("✅ Gewinner: Revisor")
    else:
        print("⚖️ Unentschieden oder unklar bewertet")

else:
    print("⚠️ Kein Goldstandard vorhanden – bitte `gold_answer` in evaluation_inputs.json eintragen.")
