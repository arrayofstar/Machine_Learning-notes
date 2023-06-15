import time

import gradio as gr
import numpy as np


# define core fn, which returns a generator {steps} times before returning the image
def fake_diffusion(steps):
    for _ in range(int(steps)):
        time.sleep(1)
        image = np.random.random((600, 600, 3))
        yield image
    image = "https://gradio-builds.s3.amazonaws.com/diffusion_image/cute_dog.jpg"
    yield image


demo = gr.Interface(fn=fake_diffusion, inputs=gr.Slider(1, 10, 3), outputs="image")

# define queue - required for generators
demo.queue()

demo.launch()
