# === evaluator.py ===

from dotenv import load_dotenv
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Initialize evaluation LLM ---
llm = ChatOpenAI(model="gpt-4")

# --- Load single-response evaluators ---
helpfulness_eval = load_evaluator(
    "criteria", llm=llm, config={"criteria": "helpfulness"}
)
correctness_eval = load_evaluator(
    "criteria", llm=llm, config={"criteria": "correctness"}
)
relevance_eval = load_evaluator("criteria", llm=llm, config={"criteria": "relevance"})
conciseness_eval = load_evaluator(
    "criteria", llm=llm, config={"criteria": "conciseness"}
)
coherence_eval = load_evaluator("criteria", llm=llm, config={"criteria": "coherence"})

# Use built-in pairwise evaluator
pairwise_eval = load_evaluator(
    "pairwise_string", llm=llm, config={"criteria": "overall"}
)


# --- Evaluation wrapper function ---
def evaluate_pairwise(question, responder, revisor):
    """
    Evaluates two responses (responder and revisor) to the same question.
    Returns a dictionary of evaluation scores and the pairwise decision.
    """
    evaluations = {
        "helpfulness_responder": helpfulness_eval.evaluate_strings(
            input=question, prediction=responder
        ),
        "helpfulness_revisor": helpfulness_eval.evaluate_strings(
            input=question, prediction=revisor
        ),
        "correctness_responder": correctness_eval.evaluate_strings(
            input=question, prediction=responder
        ),
        "correctness_revisor": correctness_eval.evaluate_strings(
            input=question, prediction=revisor
        ),
        "relevance_responder": relevance_eval.evaluate_strings(
            input=question, prediction=responder
        ),
        "relevance_revisor": relevance_eval.evaluate_strings(
            input=question, prediction=revisor
        ),
        "conciseness_responder": conciseness_eval.evaluate_strings(
            input=question, prediction=responder
        ),
        "conciseness_revisor": conciseness_eval.evaluate_strings(
            input=question, prediction=revisor
        ),
        "coherence_responder": coherence_eval.evaluate_strings(
            input=question, prediction=responder
        ),
        "coherence_revisor": coherence_eval.evaluate_strings(
            input=question, prediction=revisor
        ),
    }

    try:
        pairwise_result = pairwise_eval.evaluate_string_pairs(
            input=question, prediction=responder, prediction_b=revisor
        )

        value = pairwise_result.get("value", "").strip()
        evaluations["pairwise_winner"] = value
        evaluations["pairwise_reasoning"] = pairwise_result.get("reasoning", "")

    except Exception as e:
        evaluations["pairwise_winner"] = "Invalid"
        evaluations["pairwise_reasoning"] = f" Error: {str(e)}"

    return evaluations
