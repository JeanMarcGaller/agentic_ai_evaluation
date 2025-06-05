# === tool_executor.py ===

from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import (
    StructuredTool,  # Wraps functions to make them usable by LLMs
)
from langchain_tavily import TavilySearch  # Search tool from Tavily integration
from langgraph.prebuilt import ToolNode  # LangGraph node to execute tools in workflows

from schemas import AnswerQuestion, ReviseAnswer  # Custom tool schemas

# Initialize Tavily search tool
tavily_tool = TavilySearch(max_results=5)  # TODO: Test max_results


def run_queries(search_queries: list[str], **kwargs):
    """
    Executes a batch of search queries using Tavily.
    Only runs if the input list is non-empty.

    Args:
        search_queries (list[str]): One or more user-generated queries.

    Returns:
        list: Search results, one per query.
    """
    if not search_queries:
        return []  # If the list is empty, return nothing
    # Run each query using Tavily and return the results
    return tavily_tool.batch([{"query": query} for query in search_queries])


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
            run_queries, name=ReviseAnswer.__name__  # Tool will be named "ReviseAnswer"
        ),
    ]
)
