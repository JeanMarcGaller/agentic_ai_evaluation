# === main.py ===

import json
from load_data import get_hotpotqa_subset
from chains import first_responder, revisor
from evaluator import evaluate_pairwise
from tool_executor import execute_tools
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage


MAX_ROUNDS = 2
NUM_QUESTIONS = 2

# Daten laden
examples = get_hotpotqa_subset(num_samples=NUM_QUESTIONS)
results = []

def extract_answer(step):
    if hasattr(step, "tool_calls") and step.tool_calls:
        return step.tool_calls[0]["args"]["answer"]
    elif hasattr(step, "content") and isinstance(step.content, str):
        return step.content  # Falls LLM einfach direkt geantwortet hat
    else:
        return "(No answer found)"

for idx, ex in enumerate(examples):
    question = ex["question"]

    # Graph definieren
    builder = MessageGraph()
    builder.add_node("draft", first_responder)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("revise", revisor)
    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")

    def event_loop(state: list[BaseMessage]) -> str:
        if len(state) >= MAX_ROUNDS:
            return END
        return "execute_tools"

    builder.add_conditional_edges("revise", event_loop)
    builder.set_entry_point("draft")
    graph = builder.compile()

    print(f"\n❓ QUESTION {idx + 1}/{NUM_QUESTIONS}: {question}")
    result = graph.invoke(question)

    # Debug-Logs, um Tool-Nutzung zu überprüfen
    print(f"Responder tool used: {hasattr(result[1], 'tool_calls') and bool(result[1].tool_calls)}")
    print(f"Revisor tool used: {hasattr(result[-1], 'tool_calls') and bool(result[-1].tool_calls)}")

    responder_answer = extract_answer(result[1])
    revisor_answer = extract_answer(result[-1])

    evaluation = evaluate_pairwise(question, responder_answer, revisor_answer)

    results.append({
        "question": question,
        "responder_answer": responder_answer,
        "revisor_answer": revisor_answer,
        "evaluation": evaluation
    })

    print("✅ Evaluation completed")

# Ergebnisse speichern
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n📄 RESULTS STORED IN results.json")
