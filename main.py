# === main.py ===

"""
Application Entry point

This script evaluates two conversational agents: a responder and a revisor,
on a subset of the HotpotQA or a user-defined dataset.

Workflow:

1. Starts a local Ollama server and loads an opensource model.
2. Constructs LangGraph pipelines to answer questions, use tools, and revise responses.
3. Evaluates the two agents on the same questions and collects evaluation metrics.
4. Compares responder vs. revisor answers using pair-wise evaluation.
5. Saves results to JSON.

To process your own questions:
1. Define your questions in my_questions.json.
2. Run: python main.py --questions data/my_questions.json
"""

# === Imports ===

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, cast

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from langsmith import traceable

from chains import build_responder, build_revisor
from evaluator import evaluate_pairwise
from load_data import get_hotpotqa_subset, load_custom_questions

# Local utility modules
from ollama_manager import prepare_ollama
from tool_executor import execute_tools

# === Logging ===
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"run_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Konsole
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# === Constants ===

NUM_QUESTIONS = 1
MAX_MESSAGES = 3

OLLAMA_MODEL_NAME = "qwen3:32b"  # Ollama model to use
OPENAI_MODEL_NAME = "gpt-4.1"  # OpenAi model to use

# === Start Ollama ===

prepare_ollama(model=OLLAMA_MODEL_NAME)

# === Define model configurations ===

model_configs = {
    "ollama": ChatOllama(model=OLLAMA_MODEL_NAME),
    "openai": ChatOpenAI(model=OPENAI_MODEL_NAME),
}

# === Define model names for results recording ===

model_names = {"ollama": OLLAMA_MODEL_NAME, "openai": OPENAI_MODEL_NAME}

# === CLI Argument Parsing ===

parser = argparse.ArgumentParser()
parser.add_argument(
    "--questions",
    help="Path to my_questions.json file.",
    default=None,
)
cli_args = parser.parse_args()

# === Load Dataset ===

if cli_args.questions:
    # Load user questions from JSON
    examples = load_custom_questions(cli_args.questions)
    NUM_QUESTIONS = len(examples)
else:
    # Use a default subset of HotpotQA
    examples = get_hotpotqa_subset(num_samples=NUM_QUESTIONS)

logger.info("Loaded %s questions", NUM_QUESTIONS)

# === Results Placeholder ===

results = []

# === Extract Final Answer ===


def extract_answer(step):
    return (
        step.tool_calls[0]["args"]["answer"]
        if step.tool_calls
        else step.content if isinstance(step.content, str) else "(No answer found)"
    )


# === Evaluation ===


@traceable(name="HotpotQA Evaluation")
def evaluate_question(
    question,
    responder_answer,
    revisor_answer,
):
    # Pairwise evaluation function
    return evaluate_pairwise(
        question=question,
        responder=responder_answer,
        revisor=revisor_answer,
    )


# === Compare responder/revisor model pairs ===

model_pairs = [
    ("ollama", "ollama"),  # Compare Ollama responder vs. Ollama revisor
    ("openai", "openai"),  # Compare OpenAI responder vs. OpenAI revisor
]

# === Main Loop ===

for responder_model_name, revisor_model_name in model_pairs:
    logger.info(
        "=== Running: Responder=%s, Revisor=%s ===",
        responder_model_name,
        revisor_model_name,
    )

    # Load LLMs
    responder_llm = model_configs[responder_model_name]
    revisor_llm = model_configs[revisor_model_name]

    # Model name for results metadata
    responder_model_actual = model_names[responder_model_name]
    revisor_model_actual = model_names[revisor_model_name]

    for idx, ex in enumerate(examples):
        question = ex["question"]

        # === Build responder and revisor chains ===
        responder_chain = build_responder(responder_llm)
        revisor_chain = build_revisor(revisor_llm)

        # === Define LangGraph ===
        builder = MessageGraph()

        # Nodes / Steps
        builder.add_node("draft", responder_chain)  # Initial draft generation
        builder.add_node("execute_tools", execute_tools)  # Execute tools after draft
        builder.add_node("revise", revisor_chain)  # Final revision step

        # Edges / Transitions
        builder.add_edge("draft", "execute_tools")  # From draft to tools
        builder.add_edge("execute_tools", "revise")  # From tools to revision

        # Entry point / Start
        builder.set_entry_point("draft")  # Start from the draft ste

        def event_loop(state: List[BaseMessage]) -> str:
            # If we have reached the maximum number of steps, stop the graph
            # Otherwise, go back to execute_tools
            return END if len(state) >= MAX_MESSAGES else "execute_tools"

        # After revise, use the event_loop function to decide:
        # - to stop (END)
        # - loop back to execute_tools
        builder.add_conditional_edges("revise", event_loop)

        # Compile LangGraph pipeline
        graph = builder.compile()

        # === Execute pipeline ===
        logger.info("QUESTION %s/%s: %s", idx + 1, NUM_QUESTIONS, question)

        try:
            raw_result = graph.invoke([HumanMessage(content=question)])

        except Exception:
            logger.exception("Graph invocation failed for question: %s", question)
            continue

        result: List[BaseMessage] = cast(List[BaseMessage], raw_result)

        responder_tool_used = bool(getattr(result[1], "tool_calls", []))
        revisor_tool_used = bool(getattr(result[-1], "tool_calls", []))

        responder_answer = extract_answer(result[1])
        revisor_answer = extract_answer(result[-1])

        logger.info("Responder tool used: %s", responder_tool_used)
        logger.info("Revisor tool used: %s", revisor_tool_used)

        # === Evaluate results ===
        evaluation = evaluate_question(
            question=question,
            responder_answer=responder_answer,
            revisor_answer=revisor_answer,
        )

        # Append results
        results.append(
            {
                "question": question,
                "responder_answer": responder_answer,
                "revisor_answer": revisor_answer,
                "responder_tool_used": responder_tool_used,
                "revisor_tool_used": revisor_tool_used,
                "responder_model": responder_model_actual,
                "revisor_model": revisor_model_actual,
                "evaluation": evaluation,
            }
        )

        logger.info("Evaluation for question %s completed", idx + 1)

# === Save results ===

results_path = Path("results/results.json")
results_path.parent.mkdir(exist_ok=True)
results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
logger.info("Results stored in %s", results_path)
