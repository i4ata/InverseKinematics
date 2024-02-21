"""https://en.wikipedia.org/wiki/Inverse_kinematics#The_Jacobian_inverse_technique"""

import torch

from forward_kinematics import ForwardKinematics

from typing import Literal

class JacobianInverseTechnique:
    
    def __init__(self, fk: ForwardKinematics, h: float = .005, tolerance: float = 1e-3) -> None:
        self.fk = fk
        
        # Choose precision when calculating the jacobian. The smaller the better but slower convergence
        self.h = h
        self.tolerance = tolerance

    def _calculate_jacobian_matrix(self, angles: torch.Tensor, ee_pred: torch.Tensor, dimensions: Literal[2, 3] = 2) -> torch.Tensor:
        """Self explanatory"""
        
        # Intialize jacobian matrix (batch_size, (x,y), number_joints) for 2d and (batch_size, (x,y,z), number_joints * 2) for 3D
        jacobian_matrix = torch.zeros(size=(len(angles), dimensions, angles.shape[1]))

        # Compute the jacobian matrix by estimating the partial derivatives
        for i in range(angles.shape[1]):

            angles_copy = torch.clone(angles)
            angles_copy[:, i] += self.h
            
            jacobian_matrix[:, :, i] = (self.fk.run(angles=angles_copy, dimensions=dimensions) - ee_pred) / self.h

        return jacobian_matrix

    def run(self, ee_true: torch.Tensor, max_steps = 100) -> torch.Tensor:
        """Run the algorithm"""

        # Start off with random angles
        dimensions = ee_true.shape[1]
        angles = (torch.rand(size=(len(ee_true), len(self.fk.lengths) * (dimensions - 1))) * 2 - 1) * torch.pi
        
        # optimization loop
        for step in range(max_steps):

            # Get the predicted endeffector position
            ee_pred = self.fk.run(angles=angles, dimensions=dimensions)

            # Calculate the error
            error = ee_true - ee_pred
            norm_error = torch.linalg.norm(error, dim=1)
            mean_error = torch.mean(norm_error).item()
            print(f'{step=} | {mean_error=:.5f}')

            # If all predictions are really good, stop the loop (this might not be fully appropriate)
            not_converged_mask = norm_error > self.tolerance
            if torch.all(torch.logical_not(not_converged_mask)):
                print('Solver converged early.')
                break

            # calculate the jacobian matrix
            jacobian_matrix = self._calculate_jacobian_matrix(angles=angles[not_converged_mask], 
                                                              ee_pred=ee_pred[not_converged_mask], 
                                                              dimensions=dimensions)
            
            # Get the pinv
            jacobian_matrix_pinv = torch.linalg.pinv(jacobian_matrix)
            
            # Update the angles
            delta_angles = torch.matmul(jacobian_matrix_pinv, error[not_converged_mask].unsqueeze(-1)).view_as(angles[not_converged_mask])
            angles[not_converged_mask] += delta_angles
        
        return angles
    

if __name__ == '__main__':
    
    # Test run
    dims = 3
    n_links=6

    torch.manual_seed(0)
    fk = ForwardKinematics(n_links=n_links)
    jit = JacobianInverseTechnique(fk)
    sample_pos = fk.run(dimensions=dims, n_samples=10_000)
    angles = jit.run(sample_pos)
    pred_pos = fk.run(angles=angles, dimensions=dims)
    print(torch.mean(torch.linalg.norm(sample_pos - pred_pos, dim=1)))
    