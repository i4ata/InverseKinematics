import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from nn.model import KinematicsModel
from nn.dataset import AnglesDataset
from nn.early_stopper import EarlyStopper

from typing import Optional, Literal
from tqdm.auto import tqdm
import os
import argparse

from forward_kinematics import ForwardKinematics

seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dimensions',         type=int,   default=2,          help='Number of dimensions in the simulation')
    parser.add_argument('--name',               type=str,   default='model',    help='Name of the model')

    parser.add_argument('--batch_size',         type=int,   default=100,        help='Mini batch size for training')
    parser.add_argument('--learning_rate',      type=float, default=1e-3,       help='Optimizer learning rate')
    parser.add_argument('--epochs',             type=int,   default=50,         help='Number of epochs to train for')

    parser.add_argument('--n_hidden_layers',    type=int,   default=3,          help='Number of hidden layers of the model')
    parser.add_argument('--hidden_size',        type=int,   default=32,         help='Size of the hidden layers')

    parser.add_argument('--patience',           type=int,   default=5,          help='Early stopper patience')

    parser.add_argument('--dataset_dir',        type=str,   default='data',     help='Location of the generated dataset')
    parser.add_argument('--n_samples',          type=int,   default=int(1e6),   help='Dataset size')
    parser.add_argument('--n_links',            type=int,   default=6,          help='Number of links of the robotic arm')
    parser.add_argument('--train_size',         type=float, default=.9,         help='Proportion of data used for training')

    return parser.parse_args()

def train_step(model: KinematicsModel,
               loss_fn: nn.Module,
               optimizer: Optimizer,
               train_dataloader: DataLoader,
               fk: ForwardKinematics,
               dimensions: Literal[2, 3] = 2):
    """Standard things. Returns the mean loss for the dataloader"""
    model.train()
    train_loss = 0
    for sample, in train_dataloader:
        sample = sample.to(DEVICE)
        angles_pred = model(sample)
        ee_pred = fk.run(angles=angles_pred, dimensions=dimensions)
        loss = loss_fn(ee_pred, sample)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_dataloader)

def test_step(model: KinematicsModel,
              loss_fn: nn.Module,
              test_dataloader: DataLoader,
              fk: ForwardKinematics,
              dimensions: Literal[2, 3] = 2):
    """Standard things. Returns the mean loss of the dataloader"""
    model.eval()
    test_loss = 0
    for sample, in test_dataloader:
        sample = sample.to(DEVICE)
        with torch.inference_mode():
            angles_pred = model(sample)
            ee_pred = fk.run(angles=angles_pred, dimensions=dimensions)
            loss = loss_fn(ee_pred, sample)

        test_loss += loss.item()
    return test_loss / len(test_dataloader)

def train(model: KinematicsModel,
          loss_fn: nn.Module,
          optimizer: Optimizer,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          fk: ForwardKinematics,
          early_stopper: Optional[EarlyStopper],
          dimensions: Literal[2, 3] = 2,
          epochs: int = 10,
          save_dir: str = 'model.pt'):
    
    """Standard things"""
    for epoch in tqdm(range(epochs)):

        train_loss = train_step(model=model, loss_fn=loss_fn, optimizer=optimizer, train_dataloader=train_dataloader, fk=fk, dimensions=dimensions)
        test_loss = test_step(model=model, loss_fn=loss_fn, test_dataloader=test_dataloader, fk=fk, dimensions=dimensions)
        
        if early_stopper is not None:
            if early_stopper.check(test_loss):
                print('Training stopped early due to risk of overfitting.')
                break
            if early_stopper.save_model:
                torch.save(model, save_dir)

        print(f'{epoch=} | {train_loss=:.4f} | {test_loss=:.4f}')
    else:
        print('The training did not converge. Risk of underfitting.')
        torch.save(model.state_dict(), save_dir)

if __name__ == '__main__':

    args = parse_args()
    km = KinematicsModel(n_links=args.n_links, 
                         n_hidden_layers=args.n_links, 
                         hidden_size=args.hidden_size, 
                         dimensions=args.dimensions).to(DEVICE)
    fk = ForwardKinematics(n_links=args.n_links, device=DEVICE)
    loss = nn.MSELoss()
    early_stopper = EarlyStopper(patience=args.patience)
    optimizer = Adam(km.parameters(), lr=args.learning_rate)

    if not os.path.exists(args.dataset_dir + '.pt'):
        data = fk.run(n_samples=args.n_samples, dimensions=args.dimensions)
        # torch.save(data, args.dataset_dir + '.pt')
        print(f'Generated dataset with {args.n_samples} samples')
    else:
        data = torch.load(args.dataset + '.pt')

    dataset = AnglesDataset(data=data)
    dataset.split(train_size=args.train_size)
    dataset.get_dataloaders(batch_size=args.batch_size)

    print('Training begins')
    train(model=km, 
          loss_fn=loss, 
          optimizer=optimizer, 
          train_dataloader=dataset.train_dataloader, 
          test_dataloader=dataset.test_dataloader, 
          early_stopper=early_stopper,
          fk=fk,
          dimensions=args.dimensions,
          epochs=args.epochs,
          save_dir='models/' + args.name + '.pt')
    