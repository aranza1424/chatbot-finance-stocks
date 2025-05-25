""" Define tools for model"""
import inspect
from langchain.tools import tool
from langgraph.prebuilt import ToolNode

#function to get static methods from class

def get_tools(cls):
    return [
        func for _, func in inspect.getmembers(cls)
        if callable(func) and hasattr(func, "args_schema")
    ]


class ToolboxMain:
        @staticmethod
        @tool
        def query(update_type: str) -> str:
            """Tool to query company-related information."""
            return f"Triggered query with update_type={update_type}"


tools_main = get_tools(ToolboxMain)
tool_node_main = ToolNode(tools_main)
