# === main.py ===

import json
from load_data import get_hotpotqa_subset
from chains import first_responder, revisor
from evaluator import evaluate_pairwise
from tool_executor import execute_tools
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage, ToolMessage


MAX_ITERATIONS = 2
NUM_QUESTIONS = 1

# Subset laden
examples = get_hotpotqa_subset(num_samples=NUM_QUESTIONS, force_reload=True)

results = []

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
        num_tool_uses = sum(isinstance(msg, ToolMessage) for msg in state)
        if num_tool_uses >= MAX_ITERATIONS:
            return END
        return "execute_tools"

    builder.add_conditional_edges("revise", event_loop)
    builder.set_entry_point("draft")
    graph = builder.compile()

    print(f"\n QUESTION {idx + 1}/{NUM_QUESTIONS}: {question}")
    result = graph.invoke(question)

    responder_answer = result[1].tool_calls[0]["args"]["answer"]
    revisor_answer = result[-1].tool_calls[0]["args"]["answer"]

    evaluation = evaluate_pairwise(question, responder_answer, revisor_answer)

    results.append({
        "question": question,
        "responder_answer": responder_answer,
        "revisor_answer": revisor_answer,
        "evaluation": evaluation
    })

    print("EVALUATION COMPLETED")

# Ergebnisse speichern
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nRESULTS STORED IN results.json")
