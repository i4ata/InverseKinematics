import gradio as gr
import argparse

import torch
import matplotlib.pyplot as plt

from forward_kinematics import ForwardKinematics
from jacobian_inverse_technique import JacobianInverseTechnique
from nn.model import KinematicsModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', type=int, default=2, help='Number of dimensions of the simulation')
    parser.add_argument('--model', type=str, default='JIT', help='Which model to use')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    fk = ForwardKinematics()

    if args.model == 'DNN':
        model: KinematicsModel = torch.load(f'models/model_{args.dimensions}D.pt', map_location='cpu')
        def predict(*angles):
            with torch.inference_mode():
                predicted_angles = model(torch.FloatTensor(angles).unsqueeze(0)).squeeze(0).detach().numpy()
            fk.plot(dimensions=args.dimensions, angles=predicted_angles)
            return plt
    else:
        model = JacobianInverseTechnique(fk=fk)
        def predict(*angles):
            predicted_angles = model.run(torch.FloatTensor(angles).unsqueeze(0)).squeeze(0).numpy()
            fk.plot(dimensions=args.dimensions, angles=predicted_angles)
            return plt

    inputs = [gr.Number(label=coordinate) for dimension, coordinate in zip(range(args.dimensions), 'xyz')]
    demo = gr.Interface(fn=predict, 
                        inputs=inputs, 
                        outputs=gr.Plot(), 
                        title=f'Inverse Kinematics Using {args.model}', 
                        description=f'Inverse kinematics solver in {args.dimensions}-dimensional simulations')
    demo.launch()