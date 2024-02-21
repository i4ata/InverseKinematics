import numpy as np
import matplotlib.pyplot as plt
import torch
import math

from numpy.typing import ArrayLike
from typing import Optional, Literal

class ForwardKinematics:

    def __init__(self, n_links: Optional[int] = 6, links_lengths: Optional[ArrayLike] = None, device: str = 'cpu') -> None:

        # If the lengths are not given, initialize them with 1s
        # shape: (number_links)
        self.device = device
        self.lengths = links_lengths if links_lengths is not None else torch.ones(n_links, device=device)

    def _run_2d(self, n_samples: Optional[int] = None, angles: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Do FK in 2D. If n_samples is passed, generate that many samples, if angles are passed compute the endeffector positions"""
        
        assert n_samples is not None or angles is not None
        
        # If the angles are not given, start off with random ones (uniformly distributed between -pi and pi)
        # shape: (batch_size, number_links)
        angles = angles if angles is not None else (torch.rand(size=(n_samples, len(self.lengths)), device=self.device) * 2 - 1) * torch.pi

        # Initialize the position and angle accumulators
        x, y, theta = torch.zeros(size=(3, len(angles)), device=self.device)

        for i in range(len(self.lengths)):
            
            # Update the total angle and the position
            theta = torch.add(theta, angles[:, i])
            x = torch.add(x, self.lengths[i] * torch.cos(theta))
            y = torch.add(y, self.lengths[i] * torch.sin(theta))
        
        # shape: (batch_size, (x,y)), i.e. (batch_size, 2)
        return torch.stack((x, y), dim=1)
    
    def _run_3d(self, n_samples: Optional[int] = None, angles: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Do FK in 3D. If n_samples is passed, generate that many samples, if angles are passed compute the endeffector positions"""
        
        assert n_samples is not None or angles is not None
        
        # If the angles are not given, start off with random ones (uniformly distributed between -pi and pi)
        # shape: (batch_size, number_links, 2)
        # The last dimension is 2 because each link is characterized by 2 angles: the angle it makes with the
        # positive z-axis (theta_1) and the angle its projection over the xy plane makes with the positive x-axis (theta_2)
        angles_shape = (len(angles) if angles is not None else n_samples, len(self.lengths), 2)
        angles = angles.view(angles_shape) if angles is not None else (torch.rand(size=angles_shape, device=self.device) * 2 - 1) * torch.pi

        # Initialize the position and angle accumulators
        x, y, z, theta_1, theta_2 = torch.zeros(size=(5, len(angles)), device=self.device)
        
        for i in range(len(self.lengths)):
            
            # Update the angles
            theta_1 = torch.add(theta_1, angles[:, i, 0])
            theta_2 = torch.add(theta_2, angles[:, i, 1])

            # Calculate the sines and cosines here to reuse the values and not compute them multiple times
            theta_1_sin, theta_1_cos = torch.sin(theta_1), torch.cos(theta_1)
            theta_2_sin, theta_2_cos = torch.sin(theta_2), torch.cos(theta_2)

            # Update the position
            x = torch.add(x, self.lengths[i] * theta_1_cos * theta_2_sin)
            y = torch.add(y, self.lengths[i] * theta_1_sin * theta_2_sin)
            z = torch.add(z, self.lengths[i] * theta_2_cos)

        # shape: (batch_size, (x,y,z)), i.e. (batch_size, 3)
        return torch.stack((x, y, z), dim=1)
            
    def run(self, dimensions: Literal[2, 3] = 2, n_samples: Optional[int] = None, angles: Optional[torch.Tensor] = None) -> torch.Tensor:
        func = self._run_2d if dimensions == 2 else self._run_3d
        return func(n_samples=n_samples, angles=angles)

    def plot(self, dimensions: Literal[2, 3] = 2, angles: Optional[ArrayLike] = None, filename: Optional[str] = None) -> None:
        func = self._plot_2d if dimensions == 2 else self._plot_3d
        func(angles=angles)
        if filename is not None:
            plt.savefig(filename)

    def _plot_2d(self, angles: Optional[ArrayLike] = None) -> None:
        """Plot a single example in 2D"""

        angles = angles if angles is not None else np.random.uniform(low=-math.pi, high=math.pi, size=len(self.lengths))
        x, y, theta = 0., 0., 0.
        
        plt.figure()
        cm = plt.colormaps['tab10']
        
        for i in range(len(self.lengths)):
            theta += angles[i]
            x_i = self.lengths[i] * math.cos(theta)
            y_i = self.lengths[i] * math.sin(theta)

            plt.arrow(x, y, x_i, y_i, width=0.1, head_width=0.2, head_length=0.2, length_includes_head=True, color=cm(i), zorder=3)
            
            x += x_i
            y += y_i
        
        plt.scatter(x, y, s=50, color='red', label='End', zorder=4)
        plt.scatter(0, 0, s=50, color='black', label='Start', zorder=4)
        plt.grid(zorder=0)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title('Example of forward kinematics in 2D')
        # plt.savefig('2d_fk.pdf')
        # plt.show()

    def _plot_3d(self, angles: Optional[ArrayLike] = None) -> None:
        """Plot a single example in 3D"""

        angles_shape = (len(self.lengths), 2)
        angles = np.reshape(angles, angles_shape) if angles is not None else np.random.uniform(low=-math.pi, high=math.pi, size=angles_shape)
        x, y, z, theta_1, theta_2 = 0., 0., 0., 0., 0

        fig = plt.figure()
        cm = plt.colormaps['tab10']
        ax = fig.add_subplot(projection='3d')

        for i in range(len(self.lengths)):
            
            theta_1 += angles[i, 0]
            theta_2 += angles[i, 1]
            
            x_i = self.lengths[i] * math.cos(theta_1) * math.sin(theta_2)
            y_i = self.lengths[i] * math.sin(theta_1) * math.sin(theta_2)
            z_i = self.lengths[i] * math.cos(theta_2)

            ax.quiver(x, y, z, x_i, y_i, z_i, length=self.lengths[i].item(), color = cm(i), arrow_length_ratio=.15)

            x += x_i
            y += y_i
            z += z_i

        ax.scatter(x, y, z, s=50, color='red', label='End')
        ax.scatter(0, 0, 0, s=50, color='black', label='Start')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_title('Example of forward kinematics in 3D')
        # plt.savefig('3d_fk.pdf')
        # plt.show()

if __name__ == '__main__':
    fk = ForwardKinematics(n_links=6)
    fk.plot()
    plt.show()
    