""" Agent
"""
import os
import gradio as gr
from agent.agent_graph import AgentGraph


class ChatInterface:

    def __init__(self, graph : AgentGraph):
        
        self.my_graph = graph()
        self.demo = gr.ChatInterface(
                    self.my_graph.get_response,
                    additional_inputs=[
                        gr.Textbox(value="Me", label="thread_id"),                        
                    ],
                )

    def run(self):
        self.demo.launch()


if __name__ == "__main__":
    chat = ChatInterface(AgentGraph)
    chat.run()
