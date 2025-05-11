# === main.py ===

import json
from load_data import get_hotpotqa_subset
from chains import build_responder, build_revisor
from evaluator import evaluate_pairwise
from tool_executor import execute_tools
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Constants
MAX_ROUNDS = 2
NUM_QUESTIONS = 5

# Define responder/revisor models to compare
model_configs = {
    "ollama": ChatOllama(model="llama3.1"),
    "openai": ChatOpenAI(model="o4-mini")
}

# Load data
examples = get_hotpotqa_subset(num_samples=NUM_QUESTIONS)
results = []

def extract_answer(step):
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

# Run loop over model pairs
model_pairs = [
    ("ollama", "ollama"),
    ("openai", "openai")
]

for responder_model_name, revisor_model_name in model_pairs:
    print(f"\n=== Running: Responder={responder_model_name}, Revisor={revisor_model_name} ===")

    responder_llm = model_configs[responder_model_name]
    revisor_llm = model_configs[revisor_model_name]

    for idx, ex in enumerate(examples):
        question = ex["question"]
        gold_answer = ex.get("answer", [""])[0]

        # Build model-specific responder/revisor
        responder_chain = build_responder(responder_llm)
        revisor_chain = build_revisor(revisor_llm)

        # LangGraph
        builder = MessageGraph()
        builder.add_node("draft", responder_chain)
        builder.add_node("execute_tools", execute_tools)
        builder.add_node("revise", revisor_chain)
        builder.add_edge("draft", "execute_tools")
        builder.add_edge("execute_tools", "revise")
        builder.set_entry_point("draft")

        def event_loop(state: list[BaseMessage]) -> str:
            return END if len(state) >= MAX_ROUNDS else "execute_tools"

        builder.add_conditional_edges("revise", event_loop)
        graph = builder.compile()

        print(f"\nQUESTION {idx + 1}/{NUM_QUESTIONS}: {question}")
        result = graph.invoke(question)

        responder_tool_used = hasattr(result[1], 'tool_calls') and bool(result[1].tool_calls)
        revisor_tool_used = hasattr(result[-1], 'tool_calls') and bool(result[-1].tool_calls)

        responder_answer = extract_answer(result[1])
        revisor_answer = extract_answer(result[-1])

        print(f"Responder tool used: {responder_tool_used}")
        print(f"Revisor tool used: {revisor_tool_used}")

        evaluation = evaluate_question(
            question=question,
            responder_answer=responder_answer,
            revisor_answer=revisor_answer,
        )

        results.append({
            "question": question,
            "responder_answer": responder_answer,
            "revisor_answer": revisor_answer,
            "responder_tool_used": responder_tool_used,
            "revisor_tool_used": revisor_tool_used,
            "responder_model": responder_model_name,
            "revisor_model": revisor_model_name,
            "evaluation": evaluation,
            "gold_answer": gold_answer
        })

        print("Evaluation completed")

# Save
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults stored in: results.json")
