# === evaluator.py ===

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator

# Evaluatoren laden
llm = ChatOpenAI(model="gpt-4")
helpfulness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "helpfulness"})
relevance_eval = load_evaluator("criteria", llm=llm, config={"criteria": "relevance"})
conciseness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "conciseness"})
coherence_eval = load_evaluator("criteria", llm=llm, config={"criteria": "coherence"})
pairwise_eval = load_evaluator("pairwise_string", llm=llm, config={"criteria": "overall"})


def evaluate_pairwise(question, responder, revisor):
    # Kriterien einzeln bewerten
    evaluations = {
        "helpfulness_responder": helpfulness_eval.evaluate_strings(input=question, prediction=responder),
        "helpfulness_revisor": helpfulness_eval.evaluate_strings(input=question, prediction=revisor),
        "relevance_responder": relevance_eval.evaluate_strings(input=question, prediction=responder),
        "relevance_revisor": relevance_eval.evaluate_strings(input=question, prediction=revisor),
        "conciseness_responder": conciseness_eval.evaluate_strings(input=question, prediction=responder),
        "conciseness_revisor": conciseness_eval.evaluate_strings(input=question, prediction=revisor),
        "coherence_responder": coherence_eval.evaluate_strings(input=question, prediction=responder),
        "coherence_revisor": coherence_eval.evaluate_strings(input=question, prediction=revisor),
    }

    # Direkter Vergleich
    pairwise_result = pairwise_eval.evaluate_string_pairs(
        input=question,
        prediction=responder,
        prediction_b=revisor
    )

    evaluations["pairwise_winner"] = pairwise_result.get("value")
    evaluations["pairwise_reasoning"] = pairwise_result.get("reasoning")

    return evaluations
