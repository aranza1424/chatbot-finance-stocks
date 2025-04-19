"""For more information on `huggingface_hub` Inference API support,
please check the docs:
https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
import os
import gradio as gr
from huggingface_hub import InferenceClient
from huggingface_hub import login

from dotenv import load_dotenv


# load_dotenv(override=True)
# hf_token= os.getenv('HF_TOKEN')

# login(token = hf_token)


# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [message, history, system_message, max_tokens, temperature, top_p]

    return (
        message + str(system_message) + str(max_tokens) + str(temperature) + str(top_p)
    )


"""
For information on how to customize the ChatInterface,
peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()
