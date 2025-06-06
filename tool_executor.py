# === tool_executor.py ===

"""Wrapper for Tavily-Search, Responder/Revisor can call this tool."""

# --- Imports ---
from __future__ import annotations

import logging
from typing import List

from dotenv import load_dotenv
from langchain_core.tools import (
    StructuredTool,  # Wraps functions to make them usable by LLMs
)
from langchain_tavily import TavilySearch  # Search tool from Tavily integration
from langgraph.prebuilt import ToolNode  # LangGraph node to execute tools in workflows

from schemas import AnswerQuestion, ReviseAnswer  # Custom tool schemas

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Environment ---
load_dotenv()

# Initialize Tavily search tool
tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: List[str], **kwargs):
    """
    Executes a batch of search queries using Tavily.
    Only runs if the input list is non-empty.

    Args:
        search_queries (list[str]): One or more user-generated queries.

    Returns:
        list: Search results, one per query.
    """
    if not search_queries:
        logger.debug("run_queries: empty request, nothing to do")
        return []

    # Run each query using Tavily and return the results
    logger.info("run_queries: Start %s search requests", len(search_queries))
    results = tavily_tool.batch([{"query": q} for q in search_queries])
    logger.info("run_queries: Tavily search delivers  %s result blocks", len(results))
    return results

    # return tavily_tool.batch([{"query": query} for query in search_queries])


# Wrap run_queries into LangChain-compatible StructuredTools
execute_tools = ToolNode(
    [
        # Tool used by the responder agent
        StructuredTool.from_function(
            run_queries,
            name=AnswerQuestion.__name__,  # Tool will be named "AnswerQuestion"
        ),
        # Tool used by the revisor agent
        StructuredTool.from_function(
            run_queries,
            name=ReviseAnswer.__name__,  # Tool will be named "ReviseAnswer"
        ),
    ]
)
