# === main.py ===

import json
from load_data import get_hotpotqa_subset
from chains import first_responder, revisor
from evaluator import evaluate_pairwise
from tool_executor import execute_tools
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage

# Constants
MAX_ROUNDS = 2 # Max iterations for revision loop
NUM_QUESTIONS = 2 # Number of questions to evaluate

# Load data
examples = get_hotpotqa_subset(num_samples=NUM_QUESTIONS)
results = []

def extract_answer(step):
    """
        Extracts answer string from step output.
        Supports tool-based and direct LLM answers.
        """
    if hasattr(step, "tool_calls") and step.tool_calls:
        return step.tool_calls[0]["args"]["answer"]
    elif hasattr(step, "content") and isinstance(step.content, str):
        return step.content  # Falls LLM einfach direkt geantwortet hat
    else:
        return "(No answer found)"

# Main loop for processing each question
for idx, ex in enumerate(examples):
    question = ex["question"]

    # LangGraph workflow
    builder = MessageGraph()
    builder.add_node("draft", first_responder) # 1. Draft first answer
    builder.add_node("execute_tools", execute_tools) # 2. Use tools if necessary
    builder.add_node("revise", revisor) # 3. Revise answer

    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")

    # Conditional loop for multiple revisions
    def event_loop(state: list[BaseMessage]) -> str:
        if len(state) >= MAX_ROUNDS:
            return END
        return "execute_tools"

    builder.add_conditional_edges("revise", event_loop)
    builder.set_entry_point("draft")

    # Compile and run graph
    graph = builder.compile()

    print(f"\nQUESTION {idx + 1}/{NUM_QUESTIONS}: {question}")
    result = graph.invoke(question)

    # Debug tool calls
    print(f"Responder tool used: {hasattr(result[1], 'tool_calls') and bool(result[1].tool_calls)}")
    print(f"Revisor tool used: {hasattr(result[-1], 'tool_calls') and bool(result[-1].tool_calls)}")

    # Extract answers
    responder_answer = extract_answer(result[1])
    revisor_answer = extract_answer(result[-1])

    # Evaluate answers
    evaluation = evaluate_pairwise(question, responder_answer, revisor_answer)

    # Save answers and evaluation
    results.append({
        "question": question,
        "responder_answer": responder_answer,
        "revisor_answer": revisor_answer,
        "evaluation": evaluation
    })

    print("Evaluation completed")

# Save to file
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults stored in: results.json")
