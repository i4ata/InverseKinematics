import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from nn.model import KinematicsModel
from nn.dataset import AnglesDataset
from nn.early_stopper import EarlyStopper
from fk.forward_kinematics import ForwardKinematics

from typing import Optional, Literal
from tqdm.auto import tqdm
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

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
        ee_pos_pred = fk.run(angles=angles_pred, dimensions=dimensions)
        loss = loss_fn(ee_pos_pred, sample)

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
            ee_pos_pred = fk.run(angles=angles_pred, dimensions=dimensions)
            loss = loss_fn(ee_pos_pred, sample)

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
                torch.save(model.state_dict(), save_dir)

        print(f'{epoch=} | {train_loss=:.4f} | {test_loss=:.4f}')
    else:
        print('The training did not converge. Risk of underfitting.')
        torch.save(model.state_dict(), save_dir)

if __name__ == '__main__':

    dimensions = 3
    km = KinematicsModel(dimensions=dimensions).to(DEVICE)
    fk = ForwardKinematics()
    loss = nn.MSELoss()
    early_stopper = EarlyStopper()
    optimizer = Adam(km.parameters(), lr=.001)

    n_samples = 100_000
    data_filename = f'data_{dimensions}D.pt'
    if not os.path.exists(data_filename):
        data = fk.run(n_samples=n_samples, dimensions=dimensions)
        torch.save(data, data_filename)
        print(f'Generated dataset with {n_samples} samples')

    dataset = AnglesDataset(data_filename=data_filename)
    dataset.split()
    dataset.get_dataloaders()

    print('Training begins')
    train(model=km, 
         loss_fn=loss, 
         optimizer=optimizer, 
         train_dataloader=dataset.train_dataloader, 
         test_dataloader=dataset.test_dataloader, 
         early_stopper=early_stopper,
         fk=fk,
         dimensions=dimensions,
         epochs=50,
         save_dir=f'model_{dimensions}D.pt')