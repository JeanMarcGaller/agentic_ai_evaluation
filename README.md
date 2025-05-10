# Agentic AI Evaluation

This project implements a 🦜🕸️LangGraph-based multi-agent evaluation system for answering complex questions using
LLMs, tools (e.g. Tavily Web Search), and self-reflection.

It compares the quality of initial and revised answers using 
LLM-powered evaluators.

---

## 🔧 Features

- **Multi-agent LLM pipeline**: Responder and Revisor roles
- **Tool-augmented reasoning**: Uses Tavily web search when needed
- **Self-reflective answer generation**: Critiques and iterates on answers
- **LLM-based evaluation**: Evaluates helpfulness, relevance, coherence, and conciseness
- **Pairwise comparison**: Automatically chooses the better answer

---

## 🗂️ Project Structure

```text
agentic_ai_evaluation/
│
├── main.py # Entry point: runs question-answer-evaluation pipeline
├── chains.py # Defines LLM agents (responder and revisor)
├── tool_executor.py # Wraps Tavily search tool for LangGraph
├── evaluator.py # Uses GPT-4 to evaluate answer quality
├── schemas.py # Defines structured outputs and tool schemas
├── load_data.py # Loads and samples HotpotQA questions
└──  results.json # Output file containing evaluation results
```


---

## 🛠️ How It Works

1. **Load Questions**: A small sample from the HotpotQA dataset is loaded.
2. **Responder Agent**: Attempts to answer each question, using internal knowledge or calling the search tool.
3. **Revisor Agent**: Improves the initial answer based on the search results.
4. **Evaluation**: Both answers are evaluated along several dimensions, and a pairwise judgment is made.
5. **Save Results**: Everything is saved to `results.json`.

---

## 🧰 Technologies Used

- 🐍 Python 3.11
- 🦜🔗 [LangChain](https://python.langchain.com)
- 🦜🕸️ [LangGraph](https://langgraph.readthedocs.io)
- 🦜🔨 [LangSmith](https://docs.smith.langchain.com/)
- 🧠 OpenAI API
- 🤗 [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- 🔍 [Tavily](https://www.tavily.com)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/JeanMarcGaller/agentic_ai_evaluation.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables in a .env file and add your API keys:
```text
# OpenAI + Tavily keys
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key

# LangSmith Tracing configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic_ai_evaluation
LANGCHAIN_API_KEY=your-langsmith-api-key
```

4. Run the main script:
```bash
python main.py
```

This will:
- Sample 2 questions (can be changed via NUM_QUESTIONS)
- Run them through the agentic QA system 
- Save evaluations to results.json

---

## 🧪 Customization Ideas
- Replace HotpotQA with NaturalQuestions or WebQuestions
- Add support for more tools (e.g. calculator, Wikipedia search)
- Test different LLMs
---

## 📚 Citation

This project is inspired in part by the paper:

> Meng, Z., Dziri, N., Choudhury, M., Choi, Y., & Khashabi, D. (2023).  
> **Reflexion: Language Agents with Verbal Reinforcement Learning**.  
> arXiv: [2303.11366](https://arxiv.org/abs/2303.11366)

---

## 📄 Attribution

This project includes components adapted from:


- [LangGraph Reflexion Tutorial](https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/) by the LangChain team,  
  also used under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

- [Edan Marco](https://github.com/emarco177), used under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
  Original codebase: [LangGraph Reflexion Agent](https://github.com/emarco177/langgraph-course)

---

## 📝 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

See the [LICENSE](./LICENSE) file for full license text and attribution details for third-party code reused or adapted in this project, including:

- LangGraph Reflexion Tutorial (LangChain Team)
- LangGraph Agentic Evaluations (Edan Marco)



---

## 📬 Contact

If you are interested in this work, have questions or recommendations, feel free to contact me via GitHub or email: 

Jean-Marc Galler

[jeanmarc.galler@students.fhnw.ch](mailto:jeanmarc.galler@students.fhnw.ch)
