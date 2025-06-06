# === evaluator.py ===

"""Wrapper around LangChain evaluators."""

# === Imports ===
from __future__ import annotations

import logging
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.evaluation import EvaluatorType, load_evaluator
from langchain_openai import ChatOpenAI

# === Logging ===
logger = logging.getLogger(__name__)

# === Environment ===
load_dotenv()

# === Initialize evaluation LLM ===
llm = ChatOpenAI(model="gpt-4")  # all other models raise a warning
logger.info("Evaluation LLM initialised with model 'gpt-4'")

# === Single‑response evaluators ===
_eval_types = [
    ("helpfulness", "helpfulness"),
    ("correctness", "correctness"),
    ("relevance", "relevance"),
    ("conciseness", "conciseness"),
    ("coherence", "coherence"),
]

_single_evaluators = {
    name: load_evaluator(EvaluatorType.CRITERIA, llm=llm, config={"criteria": crit})
    for name, crit in _eval_types
}
logger.debug("Loaded single‑response evaluators: %s", list(_single_evaluators))

# === Pair‑wise evaluator ===
pairwise_eval = load_evaluator(
    EvaluatorType.PAIRWISE_STRING, llm=llm, config={"criteria": "overall"}
)
logger.debug("Loaded pair‑wise evaluator")

# --- Evaluation wrapper function ---


def evaluate_pairwise(question, responder, revisor):
    """
    Evaluates two responses (responder and revisor) to the same question.
    Returns a dictionary of evaluation scores and the pairwise decision.
    """
    logger.info("Evaluating answers for question: %.60s…", question)

    evaluations: Dict[str, Any] = {}

    # --- single‑response scores ---
    for name, evaluator in _single_evaluators.items():
        try:
            evaluations[f"{name}_responder"] = evaluator.evaluate_strings(
                input=question, prediction=responder
            )
            evaluations[f"{name}_revisor"] = evaluator.evaluate_strings(
                input=question, prediction=revisor
            )
        except Exception as exc:
            logger.exception("%s evaluator failed", name)
            evaluations[f"{name}_responder"] = {"error": str(exc)}
            evaluations[f"{name}_revisor"] = {"error": str(exc)}

    # --- pair‑wise winner ---
    try:
        pairwise_result = pairwise_eval.evaluate_string_pairs(
            input=question,
            prediction=responder,
            prediction_b=revisor,
        )

        if isinstance(pairwise_result, dict):
            evaluations["pairwise_winner"] = (
                pairwise_result.get("value") or ""
            ).strip()
            evaluations["pairwise_reasoning"] = pairwise_result.get("reasoning", "")
        else:
            # evaluate_string_pairs delivers None or unexpected type
            evaluations["pairwise_winner"] = "Invalid"
            evaluations["pairwise_reasoning"] = "No result from pairwise evaluator"

    except Exception as exc:
        logger.exception("Pair-wise evaluation failed")
        evaluations["pairwise_winner"] = "Invalid"
        evaluations["pairwise_reasoning"] = f"Error: {exc}"
