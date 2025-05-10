# === chains.py ===

import datetime
from dotenv import load_dotenv
load_dotenv() # Load environment variables

# LangChain & LangGraph imports
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Tool definitions (Pydantic schemas)
from schemas import AnswerQuestion, ReviseAnswer

# Initialize LLM
llm = ChatOpenAI(model="o4-mini") # TODO: Test other models

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
- Q: What is the capital of France? → A: "Paris." (no tool needed)
- Q: What startups raised funding in 2024 for AI in energy? → A: [Uses tool]

Be thoughtful and only use a tool if it meaningfully improves your answer.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

# === First Responder Node ===

# Add responder-specific instructions
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="""Answer the question as clearly and factually as possible (max. 150 words).

    - First try to answer using only your internal knowledge.
    - Only if you are uncertain or the question is likely to require up-to-date, external, or detailed information, suggest and call a search tool.
    - Do not speculate or guess.
    - If uncertain and no tools are used, explicitly say so."""
)

# Bind tool schema
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice=None  # Let LLM decide to use tool
)

# === Revisor Node ===

revise_instructions = """You are revising the previous answer using new context or tool results.

Guidelines:
- Use external tool results (e.g., search) only if they provide new, verifiable, or necessary information.
- If the original answer was sufficient and factually correct, refine it without tools.
- Avoid speculation, unsupported claims, and verbose language.
- Keep your revised answer under 150 words.
- Cite new facts using [n] and list references at the end.

Examples:
1. Q: Who is the current CEO of OpenAI?
   - Original: "I think it's Sam Altman."
   - Revised: "As of 2024, the CEO of OpenAI is Sam Altman. [1]"
   - [Tool used: Yes]

2. Q: What is the freezing point of water?
   - Original: "0°C"
   - Revised: "Water freezes at 0°C under standard atmospheric pressure."
   - [Tool used: No]

References:
- [1] https://example.com
- [2] https://example.com
"""
# Bind tool schema to revisor
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(
    tools=[ReviseAnswer],
    tool_choice=None # Let LLM decide to use tool
)
