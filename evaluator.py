# === evaluator.py ===

from dotenv import load_dotenv
load_dotenv() # Load environment variables

from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator

# --- Initialize evaluation LLM ---
llm = ChatOpenAI(model="gpt-4") # Other models raise a warning, GPT-4 is recommended

# --- Load evaluators ---
helpfulness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "helpfulness"})
correctness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "correctness"})
relevance_eval = load_evaluator("criteria", llm=llm, config={"criteria": "relevance"})
conciseness_eval = load_evaluator("criteria", llm=llm, config={"criteria": "conciseness"})
coherence_eval = load_evaluator("criteria", llm=llm, config={"criteria": "coherence"})

# --- Pairwise evaluator to compare two answers ---
pairwise_eval = load_evaluator("pairwise_string", llm=llm, config={"criteria": "overall"})

def evaluate_pairwise(question, responder, revisor):
    """
       Evaluates two responses (responder and revisor) to the same question.

       Returns:
           dict: A dictionary of individual scores plus a pairwise winner.
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
    # --- Compare the two answers ---
    pairwise_result = pairwise_eval.evaluate_string_pairs(
        input=question,
        prediction=responder,
        prediction_b=revisor
    )

    # Add results
    evaluations["pairwise_reasoning"] = pairwise_result.get("reasoning")
    evaluations["pairwise_winner"] = pairwise_result.get("value")

    return evaluations
