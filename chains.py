import datetime
from dotenv import load_dotenv
load_dotenv()  # Load environment variables

# LangChain & LangGraph imports
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Tool definitions (Pydantic schemas)
from schemas import AnswerQuestion, ReviseAnswer

# Output parsers
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# === Prompt template used by both responder and revisor ===

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a knowledgeable assistant.
Current time: {time}

1. Answer the question in a clear, fact-based and direct manner (max. 150 words).
2. Do not speculate or include opinions.
3. If uncertain, clearly state what is unknown or ambiguous.
4. Use tools (e.g. search) only when the answer cannot be confidently given from memory.

Examples:
- Q: What is the capital of France? → A: \"Paris.\" (no tool needed)
- Q: What startups raised funding in 2024 for AI in energy? → A: [Uses tool]

If you cannot answer with high certainty, you MUST use the tool to verify or retrieve facts.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

# === Shared revisor instructions ===
revise_instructions = """You are revising the previous answer using new context or tool results.

Guidelines:
- Use external tool results (e.g., search) if they provide new, verifiable, or necessary information.
- Avoid speculation, unsupported claims, and verbose language.
- Keep your revised answer under 150 words.
- Cite new facts using [n] and list references at the end.

Examples:
1. Q: Who is the current CEO of OpenAI?
   - Original: \"I think it's Sam Altman.\"
   - Revised: \"As of 2024, the CEO of OpenAI is Sam Altman. [1]\"
   - [Tool used: Yes]

2. Q: What is the freezing point of water?
   - Original: \"0°C\"
   - Revised: \"Water freezes at 0°C under standard atmospheric pressure.\"
   - [Tool used: No]

References:
- [1] https://example.com
- [2] https://example.com
"""

# === Builders for responder and revisor ===
def build_responder(llm):
    return actor_prompt_template.partial(
        first_instruction="""Answer the question as clearly and factually as possible (max. 150 words).

- First try to answer using only your internal knowledge.
- Only if you are uncertain or the question is likely to require up-to-date, external, or detailed information, suggest 
and call a search tool.
- Do not speculate or guess.
- If uncertain and no tools are used, explicitly say so."""
    ) | llm.bind_tools(
        tools=[AnswerQuestion],
        tool_choice=None
    )

def build_revisor(llm):
    return actor_prompt_template.partial(
        first_instruction=revise_instructions
    ) | llm.bind_tools(
        tools=[ReviseAnswer],
        tool_choice=None
    )

# === Available LLM instances ===
openai_llm = ChatOpenAI(model="gpt-4o")
ollama_llm = ChatOllama(model="llama3.1")

# === Default nodes (can be overridden in main.py) ===
first_responder = build_responder(openai_llm)
revisor = build_revisor(openai_llm)
