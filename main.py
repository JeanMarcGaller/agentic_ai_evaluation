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
OPENAI_MODEL_NAME = "gpt-4.1"       # OpenAi model to use: gpt-4.1

# === Start Ollama backend ===

prepare_ollama(model=OLLAMA_MODEL_NAME)


# === Define model configurations ===

model_configs = {
    "ollama": ChatOllama(model=OLLAMA_MODEL_NAME),
    "openai": ChatOpenAI(model=OPENAI_MODEL_NAME)
}

# === Define model names for results recording ===

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
    # Load user questions from JSON
    examples = load_custom_questions(cli_args.questions)
    NUM_QUESTIONS = len(examples)                         # Update number of questions
else:
    # Use a default subset of HotpotQA
    examples = get_hotpotqa_subset(num_samples=NUM_QUESTIONS)


# === Results Placeholder ===

results = []    # List to accumulate all results and metadata


# === Extract Final Answer ===

def extract_answer(step):
    """
    Extracts the final answer from a graph node step result.
    Checks for tool call outputs or direct message content.
    """
    if hasattr(step, "tool_calls") and step.tool_calls:
        return step.tool_calls[0]["args"]["answer"] # Extract answer from tool call output
    elif hasattr(step, "content") and isinstance(step.content, str):
        return step.content # Extract from message string
    else:
        return "(No answer found)"


# === Evaluation ===

@traceable(name="HotpotQA Evaluation")
def evaluate_question(question, responder_answer, revisor_answer, responder_tool_used, revisor_tool_used):
    # Pairwise evaluation function comparing two responses
    return evaluate_pairwise(
        question=question,
        responder=responder_answer,
        revisor=revisor_answer,
        # responder_tool_used=responder_tool_used,
        # revisor_tool_used=revisor_tool_used
    )

# === Compare responder/revisor model pairs ===

model_pairs = [
    ("ollama", "ollama"), # Compare Ollama responder vs. Ollama revisor
    ("openai", "openai"), # Compare OpenAI responder vs. OpenAI revisor
]


# === Main Loop ===

for responder_model_name, revisor_model_name in model_pairs:
    print(f"\n=== Running: Responder={responder_model_name}, Revisor={revisor_model_name} ===")

    # Load LLMs
    responder_llm = model_configs[responder_model_name]
    revisor_llm = model_configs[revisor_model_name]

    # Model name for results metadata
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

        # Nodes / Steps
        builder.add_node("draft", responder_chain)          # Initial draft generation
        builder.add_node("execute_tools", execute_tools)    # Execute tools after draft
        builder.add_node("revise", revisor_chain)           # Final revision step

        # Edges / Transitions
        builder.add_edge("draft", "execute_tools")      # From draft to tools
        builder.add_edge("execute_tools", "revise")     # From tools to revision

        # Entry point / Start
        builder.set_entry_point("draft")    # Start from the draft step

        # Conditional: This function decides whether to stop the graph or go for another round
        def event_loop(state: list[BaseMessage]) -> str:
            # If we have reached the maximum number of steps (e.g. 3), stop the graph
            # Otherwise, go back to the "execute_tools" step and continue
            return END if len(state) >= MAX_ROUNDS else "execute_tools"

        # After the "revise" step, use the event_loop function to decide:
        # - whether to stop (END)
        # - or loop back to "execute_tools"
        builder.add_conditional_edges("revise", event_loop)

        # Compile LangGraph pipeline
        graph = builder.compile()


        # === Execute pipeline ===
        print(f"\nQUESTION {idx + 1}/{NUM_QUESTIONS}: {question}")
        result = graph.invoke([HumanMessage(content=question)]) # Run LangGraph pipeline

        # Check if each agent used a tool during execution
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

        # Append structured result to results list
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
