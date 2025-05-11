""" Agent
"""
import os
import gradio as gr
from agent.agent_graph import AgentGraph

my_graph = AgentGraph()


def respond(message, history, thread_id):
    response = my_graph.get_response(message, thread_id=thread_id)
    return response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="Me", label="thread_id"),
        
    ],
)


if __name__ == "__main__":
    demo.launch()
