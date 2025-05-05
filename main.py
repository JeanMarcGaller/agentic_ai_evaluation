import json
from langgraph.graph import END, MessageGraph
from langchain_core.messages import BaseMessage, ToolMessage

from chains import first_responder, revisor
from tool_executor import execute_tools

# Frage formulieren
QUESTION = "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."

# Graph definieren
MAX_ITERATIONS = 2
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

# Graph ausführen
graph = builder.compile()
result = graph.invoke(QUESTION)

# Ergebnisse extrahieren
responder_answer = result[1].tool_calls[0]["args"]["answer"]
revisor_answer = result[-1].tool_calls[0]["args"]["answer"]

# Für Evaluation speichern
evaluation_data = {
    "question": QUESTION,
    "responder_answer": responder_answer,
    "revisor_answer": revisor_answer,
    "gold_answer": "An AI-powered Security Operations Center (SOC) is a centralized "
                   "unit that integrates artificial intelligence (AI) and machine "
                   "learning (ML) to detect, analyze, and respond to security threats "
                   "in real time. By automating routine tasks such as monitoring "
                   "network activity, analyzing security incidents, and prioritizing "
                   "threats, an AI-powered SOC enables faster and more efficient "
                   "responses to cyberattacks. This reduces the burden on human "
                   "analysts and improves the overall security posture of the "
                   "organization."
}

with open("evaluation_inputs.json", "w") as f:
    json.dump(evaluation_data, f, indent=2)

print("✅ Antworten gespeichert in evaluation_inputs.json")
print("\n--- Responder Answer ---\n", responder_answer)
print("\n--- Revisor Answer ---\n", revisor_answer)
