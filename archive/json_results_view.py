import json
import pandas as pd

with open("../results.json") as f:
    data = json.load(f)

# Flachstruktur für tabellarische Darstellung extrahieren
rows = []
for item in data:
    row = {
        "Question": item["question"],
        "Responder": item["responder_answer"][:200] + "...",  # gekürzt
        "Revisor": item["revisor_answer"][:200] + "...",
        "Winner": item["evaluation"].get("pairwise_winner", "-"),
        "Helpfulness R": item["evaluation"]["helpfulness_responder"]["value"],
        "Helpfulness V": item["evaluation"]["helpfulness_revisor"]["value"],
        "Relevance R": item["evaluation"]["relevance_responder"]["value"],
        "Relevance V": item["evaluation"]["relevance_revisor"]["value"],
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.head()
