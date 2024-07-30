import gradio as gr
from rag_prompt import prompt

demo = gr.Interface(fn=prompt, inputs="text", outputs="text")

demo.launch()