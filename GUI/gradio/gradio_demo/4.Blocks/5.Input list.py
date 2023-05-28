import gradio as gr

with gr.Blocks() as demo:
    a = gr.Number(label="a")
    b = gr.Number(label="b")
    with gr.Row():
        add_btn = gr.Button("Add")
        sub_btn = gr.Button("Subtract")
    c = gr.Number(label="sum")

    def add(num1, num2):
        return num1 + num2
    add_btn.click(add, inputs=[a, b], outputs=c) #要么是列表，要么是字典

    def sub(data):
        return data[a] - data[b]
    sub_btn.click(sub, inputs={a, b}, outputs=c)


demo.launch()