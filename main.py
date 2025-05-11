# === main.py ===

import json
from load_data import get_hotpotqa_subset
from chains import first_responder, revisor
from evaluator import evaluate_pairwise
from tool_executor import execute_tools
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage
from langsmith import traceable

# Constants
MAX_ROUNDS = 2  # Max iterations for revision loop
NUM_QUESTIONS = 3  # Number of questions to evaluate

# Load data
examples = get_hotpotqa_subset(num_samples=NUM_QUESTIONS)
results = []

def extract_answer(step):
    """Extracts answer string from a LangGraph step output."""
    if hasattr(step, "tool_calls") and step.tool_calls:
        return step.tool_calls[0]["args"]["answer"]
    elif hasattr(step, "content") and isinstance(step.content, str):
        return step.content
    else:
        return "(No answer found)"

@traceable(name="HotpotQA Evaluation")
def evaluate_question(question, responder_answer, revisor_answer):
    return evaluate_pairwise(
        question=question,
        responder=responder_answer,
        revisor=revisor_answer,
    )

# Main loop
for idx, ex in enumerate(examples):
    question = ex["question"]
    gold_answer = ex.get("answer", [""])[0]  # Optional, currently unused

    # LangGraph workflow setup
    builder = MessageGraph()
    builder.add_node("draft", first_responder)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("revise", revisor)
    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")

    def event_loop(state: list[BaseMessage]) -> str:
        return END if len(state) >= MAX_ROUNDS else "execute_tools"

    builder.add_conditional_edges("revise", event_loop)
    builder.set_entry_point("draft")

    # Execute
    graph = builder.compile()
    print(f"\nQUESTION {idx + 1}/{NUM_QUESTIONS}: {question}")
    result = graph.invoke(question)

    # Tool usage and extraction
    responder_tool_used = hasattr(result[1], 'tool_calls') and bool(result[1].tool_calls)
    revisor_tool_used = hasattr(result[-1], 'tool_calls') and bool(result[-1].tool_calls)
    responder_answer = extract_answer(result[1])
    revisor_answer = extract_answer(result[-1])

    print(f"Responder tool used: {responder_tool_used}")
    print(f"Revisor tool used: {revisor_tool_used}")

    # Evaluation with automatic tracing
    evaluation = evaluate_question(
        question=question,
        responder_answer=responder_answer,
        revisor_answer=revisor_answer,
    )

    # Store
    results.append({
        "question": question,
        "responder_answer": responder_answer,
        "revisor_answer": revisor_answer,
        "responder_tool_used": responder_tool_used,
        "revisor_tool_used": revisor_tool_used,
        "evaluation": evaluation,
        "gold_answer": gold_answer # Optional, currently unused
    })

    print("Evaluation completed")

# Save
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults stored in: results.json")
