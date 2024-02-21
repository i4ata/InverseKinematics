import gradio as gr
import argparse
from typing import Literal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', type=Literal[2, 3], default=2, help='Number of dimensions of the simulation')
    return parser.parse_args()

def f(model, *coords):
    return coords, model

if __name__ == '__main__':
    
    args = parse_args()

    model_options = ('Jacobian Inverse Technique', 'Deep Neural Network')
    inputs = [gr.Number(label=coordinate) for dimension, coordinate in zip(range(args.dimensions), 'xyz')]
    inputs.insert(0, gr.Radio(choices=model_options))
    demo = gr.Interface(fn=f, inputs=inputs, outputs='text')
    demo.launch()