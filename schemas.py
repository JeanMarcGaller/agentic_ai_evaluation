# === schemas.py ===

"""
Typed tool-IO schemas for LangChain agents:
- AnswerQuestion with self-reflection & search queries
- ReviseAnswer extension
"""

from typing import List

from pydantic import BaseModel, Field

# --- Helper Schema for Feedback / Self-Reflection ---


class Reflection(BaseModel):
    """
    A structured critique of an answer.
    - missing: What information is lacking?
    - superfluous: What content is unnecessary or misleading?
    """

    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


# --- Schema for Initial Answer Generation ---


class AnswerQuestion(BaseModel):
    """
    Tool output format for answering a question.
    Includes answer, self-reflection, and search intent.
    """

    answer: str = Field(
        description="~150 word detailed answer to the question."  # Main answer content
    )
    reflection: Reflection = Field(
        description="Your reflection on the initial answer."  # Self-evaluation
    )
    search_queries: List[str] = Field(
        # Model should generate helpful search queries
        description=(
            "1-3 search queries for researching "
            "improvements to address the critique "
            "of your current answer."
        )
    )


# --- Schema for Revising an Answer ---
class ReviseAnswer(AnswerQuestion):
    """
    Extension of AnswerQuestion with reference support.
    Used by the revisor node to update and cite answers.
    """

    references: List[str] = Field(
        # Sources backing up the revised content
        description="Citations motivating your updated answer."
    )
