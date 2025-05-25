""" Module to create agent graph"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4
from IPython.display import Image, display

from agent.utils.states import OverallQueryState
from agent.utils.nodes import main_nodes, query_nodes
from agent.utils.tools import tool_node_main



#create Parent class
class Agent:
    """ Parent class for Agents"""

    def show_graph(self) -> None:
        """Display the compiled graph as an image in Jupyter"""
        img_data = self.graph_compile.get_graph().draw_mermaid_png()
        display(Image(img_data))

    def get_simple_response(self, user_input: str, thread_id: str = "default") -> str:
        """Get a response"""
        try:

            config = {"configurable": {"thread_id": thread_id}}
            messages = [HumanMessage(content=user_input)]

            final_state = self.graph_compile.invoke({self.state_output: messages}, config)
            return final_state["messages"][-1].content
        except Exception as e:
            return f"There was an error calling the graph. \n {e}"


#Create subclass for queries
class AgentQuery(Agent):
    """ Agent to make queries"""
    
    def __init__(self):

        self.state_output ="question"

        graph = StateGraph(OverallQueryState)
        graph.add_node("generate_questions", query_nodes.generate_questions)
        graph.add_node("generate_query", query_nodes.generate_query)
        graph.add_node("get_final_query_response", query_nodes.get_final_query_response)
        graph.add_edge(START, "generate_questions")
        graph.add_conditional_edges("generate_questions", query_nodes.continue_to_query, ["generate_query"])
        graph.add_edge("generate_query", "get_final_query_response")
        graph.add_edge("get_final_query_response", END)

        # Compile the graph
        memory = MemorySaver()
        self.graph = graph
        self.graph_compile = graph.compile(checkpointer=memory)



#create instances for main agent

agent_query = AgentQuery()

class AgentGraph(Agent):

    def __init__(self) -> None:
        
        self.state_output ="messages"

        builder = StateGraph(OverallQueryState)

        builder.add_node("decide_tool", main_nodes.decide_tool)
        builder.add_node("execute_tools", tool_node_main)
        builder.add_node("generate_query", agent_query.graph.compile())
        
        builder.add_edge(START, "decide_tool")
        builder.add_conditional_edges("decide_tool", main_nodes.route_message)
        builder.add_edge("execute_tools", "generate_query")
        builder.add_edge("generate_query", END)

        # Checkpointer for short-term (within-thread) memory
        within_thread_memory = MemorySaver()
        
        self.graph = builder
        self.graph_compile = builder.compile(checkpointer=within_thread_memory)

    def get_gradio_response(self, user_input: str, history = None , thread_id = None) -> str:

        try:
            if thread_id is None:
                thread_id = str(uuid4())

            config = {"configurable": {"thread_id": thread_id}}
            messages = [HumanMessage(content=user_input)]

            final_state = self.graph_compile.invoke({"messages": messages}, config)
            return final_state["messages"][-1].content
        except Exception as e:
            return f"There was an error calling the graph. \n {e}"


        
if __name__ == "__main__":
    main_agent = AgentGraph()
