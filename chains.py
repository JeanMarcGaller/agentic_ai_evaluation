# === chains.py ===

"""
Factory functions that build the responder and revisor LangChain agents.
Both share one prompt template but bind different tool schemas.
"""

import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

# Parses tool output
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


# === Prompt template used by both responder and revisor ===

# Defines format and behavior for messages sent to LLM
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        # System message
        (
            "system",
            """You are a knowledgeable assistant.
Current time: {time}

Answer the question in a clear, fact-based and direct manner (max. 150 words).
Do not speculate or include opinions. 
If uncertain, clearly state what is unknown or ambiguous.
Use tools (e.g. search) when the answer cannot be
confidently given from memory.

Examples:
- What is the capital of France? \"Paris.\" (no tool needed)
- What startups raised funding in 2025 for AI in energy? (tool needed, use tool)

If you are not fully confident in your answer,
YOU MUST use the tool to verify or obtain the necessary
information.""",
        ),
        # Placeholder for conversation history
        MessagesPlaceholder(variable_name="messages"),
        # Final instruction
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

# === Shared revisor instructions ===

# Instructions for how the revisor should rewrite an answer
revise_instructions = """You are revising the previous answer using
new context or tool results.
- YOU MUST use the tool results (e.g., search)
if they provide new or necessary information.
- Avoid speculation, unsupported claims, and verbose language.
- Keep your revised answer under 150 words.
- Cite new facts using [n] and list references at the end.
Examples:
1. Q: Who is the current CEO of OpenAI?
   - Original: \"I think it's Sam Altman.\"
   - Revised: \"As of 2025, the CEO of OpenAI is Sam Altman. [1]\"
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


# Creates the responder agent
def build_responder(llm):
    return (
        actor_prompt_template.partial(
            # Additional instruction for responder
            first_instruction="""Answer the question as clearly and factually
            as possible (max. 150 words).
            - First try to answer using only your internal knowledge.
            - If you are uncertain or the question is likely to require
            up-to-date, external, or detailed information,
            YOU MUST use the tool.
            """
        )
        | llm.bind_tools(
            tools=[AnswerQuestion],  # Tool the responder can use
            tool_choice=None,  # Let the model choose when to use the tool
        )
    )


# Creates the revisor agent
def build_revisor(llm):
    return actor_prompt_template.partial(
        first_instruction=revise_instructions  # Uses revisor-specific behavior
    ) | llm.bind_tools(
        tools=[ReviseAnswer],  # Tool the revisor can use
        tool_choice=None,  # Let the model decide when to use it
    )
