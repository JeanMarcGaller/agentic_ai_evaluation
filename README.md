# Agentic AI Evaluation

This project implements a 🦜🕸️LangGraph-based multi-agent evaluation system for answering questions using
LLMs, tools (e.g., Tavily Web Search), and self-reflection.

It compares the quality of initial and revised answers using LLM as a Judge evaluators.

---

## 🔧 Features

- 🧞‍♂️ **Responder** and 🧞 **Revisor** agents in a multi-agent LLM pipeline
- 🔍 **Tool-augmented reasoning** using Tavily Web Search when internal knowledge is insufficient
- 🪞 **Self-reflective answering**, where agents critique and iteratively refine their responses
- ⚖️ **LLM-as-a-judge** evaluating answers based on helpfulness, relevance, coherence, and conciseness
- 🤝 **Pairwise comparison** to determine which answer is better overall
- 📈 **LangSmith tracing** for transparent run-level debugging and rich execution analytics  

---

## 🗂️ Project Structure

```text
agentic_ai_evaluation/
│
├── main.py # Entry point: runs question-answer-evaluation pipeline
├── ollama_manager.py # Setup of Ollama backend
├── load_data.py # Loads and samples questions
├── chains.py # Defines LLM agents (responder and revisor)
├── schemas.py # Defines structured outputs and tool schemas
├── tool_executor.py # Wraps Tavily search tool for LangGraph
├── evaluator.py # Uses GPT-4o-mini to evaluate answer quality
├── results/
│   ├── results.ipynb # Notebook to view results
│   └── results.json # Output file containing evaluation results
├── data/
│   ├── hotpotqa_subset_20250101_010101.json # HotpotQA Sample questions
│   └── my_questions.json # Custom dataset with own questions
└── log/
    └── run20250101_010101.log # Log files
```

---

## 🛠️ How It Works

1. 📥 **Load questions** from a small sample of the HotpotQA dataset
2. 🧞‍♂️ **Responder agent** generates an initial answer using internal knowledge or Tavily web search
3. 🧞 **Revisor agent** critiques and improves the initial response using new context or tool results
4. ⚖️ **LLM evaluator** scores both answers on multiple criteria and performs a pairwise comparison
5. 💾 **Save results** to `results.json` for analysis or reporting
---

## 🧰 Technologies Used

- 🐍 [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- 🦜🔗 [LangChain](https://python.langchain.com)
- 🦜🕸️ [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
- 🦜🔨 [LangSmith](https://docs.smith.langchain.com/)
- 🧠  [OpenAI API](https://openai.com/index/openai-api/)
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

Or with your own questions, stored in my_questions.json:

```bash
python main.py --questions data/my_questions.json
```

Before committing, **run**:  
```bash
black .
```  
Then run `pre-commit run --all-files` if you have the hooks installed.  


## ⚙️ Configuration Parameters

These constants let you fine-tune a run without touching the core code.  
You can override them via environment variables or CLI flags if needed.

| Name                | Default | Meaning                                                                                              |
|---------------------|---------|------------------------------------------------------------------------------------------------------|
| `MAX_MESSAGES`      | `3`     | Hard stop for a single QA turn. One “round” consists of **3 messages** (user ➞ responder ➞ revisor). |
| `NUM_QUESTIONS`     | `5`     | How many questions are sampled and evaluated per execution of `main.py`.                             |
| `OLLAMA_MODEL_NAME` | `qwen3:32b` | Local **Ollama** model used by the responder/revisor agents. Choose model you have pulled.           |
| `OPENAI_MODEL_NAME` | `gpt-4.1` | Remote **OpenAI** model used by the responder/revisor agents.                                        |


## 📊 Example Results

| Question                                                                                                             | Responder Tool | Revisor Tool | Winner    | Helpfulness Responder | Helpfulness Revisor | Correctness Responder | Correctness Revisor | Relevance Responder | Relevance Revisor | Conciseness Responder | Conciseness Revisor | Coherence Responder | Coherence Revisor |
|----------------------------------------------------------------------------------------------------------------------|----------------|--------------|-----------|-----------------------|---------------|---------------|---------------|-------------|-------------|---------------|---------------|-------------|-------------|
| Who was appointed to the board of supervisors first, Jeff Sheehy or Ed Lee?                                          | N              | N            | Responder | Y                     | Y             | Y             | Y             | Y           | Y           | Y             | Y             | Y           | Y           |
| Who was the electoral division that James Tully represented in 1928 named after?                                     | Y              | N            | Revisor   | Y                     | Y             | Y             | Y             | Y           | Y           | Y             | Y             | Y           | Y           |
| At what school is the individual who was awarded the 2012 Nobel Prize in Physics a professor?                        | N              | N            | None      | Y                     | Y             | Y             | Y             | Y           | Y           | Y             | Y             | Y           | Y           |
| Who played the female lead in a 2007 Indian Telugu film...?                                                          | Y              | N            | Revisor   | N                     | Y             | N             | Y             | N           | Y           | N             | Y             | N           | Y           |


---

## ⚠️ Known Issues & Limitations
- High latency due to the responder–revisor cycle and the LLM-as-a-Judge evaluation
- Evaluation is expensive  
- In most cases, the initial response is already quite good
- Dataset questions are often generic, making them hard to interpret
- Yes/No evaluators offer limited insight; a graded score would likely be more informative, 
but attempts to implement such scoring have so far been unsuccessful






---

## 🧪 Customization Ideas
- Replace HotpotQA with NaturalQuestions or WebQuestions
- Add support for more tools (e.g., arXiv API, Wikipedia search)
- Implement AsyncOpenAI-Client and retry mechanism
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
 used under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

- [Edan Marco](https://github.com/emarco177), used under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
  Original codebase: [LangGraph Reflexion Agent](https://github.com/emarco177/langgraph-course)

---

## 📝 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
See the [LICENSE](./LICENSE) file for full license text and attribution details for third-party code reused or adapted in this project, including:
- LangGraph Reflexion Tutorial (LangChain Team)
- LangGraph Reflexion Agent (Edan Marco)
---

## 📬 Contact
This repository is part of an academic project and is provided for demonstration and review purposes only.  
If you are interested in this work, have questions or recommendations, feel free to contact me via email: 

Jean-Marc Galler

📧 jeanmarc.galler [at] students [dot] fhnw [dot] ch
