# === tool_executor.py ===

from dotenv import load_dotenv
load_dotenv() # Load environment variables

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

# Initialize Tavily search tool
tavily_tool = TavilySearch(max_results=5) # TODO: Test max_results

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
        return []
    return tavily_tool.batch([{"query": query} for query in search_queries])

# Wrap run_queries into LangChain-compatible StructuredTools
# Bind the tool to each schema name so the LLM can invoke it correctly
execute_tools = ToolNode([
    StructuredTool.from_function(
        run_queries, name=AnswerQuestion.__name__ # Tool callable when responder suggests it
    ),
    StructuredTool.from_function(
        run_queries,
        name=ReviseAnswer.__name__ # Tool callable when responder suggests it
    ),
])
