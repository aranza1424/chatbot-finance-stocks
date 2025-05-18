""" Module to create agent graph"""

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4

from agent.utils.tools import tools
from agent.utils.states import MessagesState
from agent.utils.nodes import nodes



class AgentGraph:
    def __init__(self):
        # Graph
        builder = StateGraph(MessagesState)

        # Define nodes: these do the work
        builder.add_node("assistant", nodes.assistant)
        builder.add_node("tools", ToolNode(tools))

        # Define edges: these determine how the control flow moves
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory)
    
    def get_response(self, user_input: str, history: list = None , thread_id: str = None) -> str:

        try:
            if thread_id is None:
                thread_id = str(uuid4())

            config = {"configurable": {"thread_id": thread_id}}
            messages = [HumanMessage(content=user_input)]

            final_state = self.graph.invoke({"messages": messages}, config)
            return final_state["messages"][-1].content
        except Exception as e:
            return f"There was an error calling the graph. \n {e.__class__.__name__,}"
        
if __name__ == "__main__":
    AgentGraph()
