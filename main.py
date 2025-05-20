# === main.py ===

"""
Application Entry point

This script evaluates two conversational agents: a responder and a revisor,
on a subset of the HotpotQA or a user-defined dataset of questions.

Workflow:
1. Starts a local Ollama server and loads an opensource model.
2. Constructs LangGraph pipelines to answer questions, use tools, and revise responses.
3. Evaluates the two agents on the same questions and collects evaluation metrics.
4. Compares responder vs. revisor answers using pair-wise evaluation.
5. Saves results.

To process your own questions:
1. Define your questions in my_questions.json.
2. Run: python main.py --questions my_questions.json
"""

# === Imports ===

import json
import argparse

# Local utility modules
from ollama_manager import prepare_ollama
from load_data import get_hotpotqa_subset, load_custom_questions
from chains import build_responder, build_revisor
from evaluator import evaluate_pairwise
from tool_executor import execute_tools

# LangGraph framework for building pipelines
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langsmith import traceable

# LLM interfaces
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


# === Constants ===

MAX_ROUNDS = 3                      # Max graph steps before stopping (used in conditional edge logic)
NUM_QUESTIONS = 10                  # Default number of questions to evaluate
OLLAMA_MODEL_NAME = "qwen3:32b"     # Ollama model to use: qwen2.5:72b, qwen3:32b
OPENAI_MODEL_NAME = "gpt-4.1"       # OpenAi model to use

# === Start Ollama backend ===

prepare_ollama(model=OLLAMA_MODEL_NAME) # Launches Ollama and ensures model is ready


# === Define model configurations ===

model_configs = {
    "ollama": ChatOllama(model=OLLAMA_MODEL_NAME),
    "openai": ChatOpenAI(model=OPENAI_MODEL_NAME)
}

model_names = {
    "ollama": OLLAMA_MODEL_NAME,
    "openai": OPENAI_MODEL_NAME
}

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
    examples = load_custom_questions(cli_args.questions)  # Load user-provided questions
    NUM_QUESTIONS = len(examples)                         # Update number of questions
else:
    examples = get_hotpotqa_subset(num_samples=NUM_QUESTIONS)  # Load subset of HotpotQA


# === Results Placeholder ===

results = [] # Stores per-question results and metadata


# === Extract Final Answer ===

def extract_answer(step):
    if hasattr(step, "tool_calls") and step.tool_calls:
        return step.tool_calls[0]["args"]["answer"]
    elif hasattr(step, "content") and isinstance(step.content, str):
        return step.content
    else:
        return "(No answer found)"


# === Evaluation ===

@traceable(name="HotpotQA Evaluation")
def evaluate_question(question, responder_answer, revisor_answer, responder_tool_used, revisor_tool_used):
    return evaluate_pairwise(
        question=question,
        responder=responder_answer,
        revisor=revisor_answer,
        # responder_tool_used=responder_tool_used,
        # revisor_tool_used=revisor_tool_used
    )

# === Compare responder/revisor model pairs ===

model_pairs = [
    ("ollama", "ollama"),
    ("openai", "openai"),
]

for responder_model_name, revisor_model_name in model_pairs:
    print(f"\n=== Running: Responder={responder_model_name}, Revisor={revisor_model_name} ===")

    # Load LLMs
    responder_llm = model_configs[responder_model_name]
    revisor_llm = model_configs[revisor_model_name]

    # Use consistent model naming
    responder_model_actual = model_names[responder_model_name]
    revisor_model_actual = model_names[revisor_model_name]

    for idx, ex in enumerate(examples):
        question = ex["question"]
        gold_answer = ex.get("answer", [""])[0] # For comparison later, unused. #TODO Check gold_answer usage in code

        # === Build responder and revisor chains ===
        responder_chain = build_responder(responder_llm)
        revisor_chain = build_revisor(revisor_llm)

        # === Define LangGraph ===
        builder = MessageGraph()

        # Nodes
        builder.add_node("draft", responder_chain)
        builder.add_node("execute_tools", execute_tools)
        builder.add_node("revise", revisor_chain)

        # Edges
        builder.add_edge("draft", "execute_tools")
        builder.add_edge("execute_tools", "revise")

        # Entry point
        builder.set_entry_point("draft")

        # Conditional: After "revise", continue or end?
        def event_loop(state: list[BaseMessage]) -> str:
            return END if len(state) >= MAX_ROUNDS else "execute_tools"

        builder.add_conditional_edges("revise", event_loop)

        # Compile LangGraph pipeline
        graph = builder.compile()


        # === Execute pipeline ===
        print(f"\nQUESTION {idx + 1}/{NUM_QUESTIONS}: {question}")
        result = graph.invoke([HumanMessage(content=question)])

        # Extract tool usage metadata
        responder_tool_used = hasattr(result[1], 'tool_calls') and bool(result[1].tool_calls)
        revisor_tool_used = hasattr(result[-1], 'tool_calls') and bool(result[-1].tool_calls)

        # Extract answers
        responder_answer = extract_answer(result[1])
        revisor_answer = extract_answer(result[-1])

        print(f"Responder tool used: {responder_tool_used}")
        print(f"Revisor tool used: {revisor_tool_used}")

        # === Evaluate results ===
        evaluation = evaluate_question(
            question=question,
            responder_answer=responder_answer,
            revisor_answer=revisor_answer,
            responder_tool_used=responder_tool_used,
            revisor_tool_used=revisor_tool_used
        )

        results.append({
            "question": question,
            "responder_answer": responder_answer,
            "revisor_answer": revisor_answer,
            "responder_tool_used": responder_tool_used,
            "revisor_tool_used": revisor_tool_used,
            "responder_model": responder_model_actual,
            "revisor_model": revisor_model_actual,
            "evaluation": evaluation,
            "gold_answer": gold_answer
        })

        print("Evaluation completed")


# === Save results ===

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults stored in: results.json")
