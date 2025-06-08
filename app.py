""" Agent
"""
import os
from dotenv import load_dotenv
import gradio as gr
from agent.agent_graph import AgentGraph

load_dotenv(override=True)


main_agent = AgentGraph()

class ChatInterface:

    def __init__(self, graph : AgentGraph):
        
        self.my_graph = graph
        self.demo = gr.ChatInterface(
                    self.my_graph.get_gradio_response,
                    additional_inputs=[
                        gr.Textbox(value="Me", label="thread_id"),                        
                    ],
                )

    def run(self) -> None:
        self.demo.launch()


if __name__ == "__main__":
    chat = ChatInterface(main_agent)
    chat.run()
