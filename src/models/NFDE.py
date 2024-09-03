from src.utils.FDE_Solver_optimized import FDEint
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import IVPRegressionModel
from tqdm import tqdm
import Constants as const

class FDE(nn.Module):
    """Trainable Fractional Differential Equation"""

    def __init__(self, dim_x, dim_z, hidden_size,
                 n_layers_hidden=3, activation='relu', include_time=True,homogeneous=False):
        super().__init__()
        self.include_time = include_time
        self.homogeneous = homogeneous
        self.layers = nn.ModuleList()
        time_dim = 1 if include_time else 0
        if not homogeneous:
            self.layers.append(nn.Linear(time_dim + dim_x + dim_z, hidden_size))
        else:
            self.layers.append(nn.Linear(time_dim + dim_z, hidden_size))

        for _ in range(n_layers_hidden - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, dim_z))

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise ValueError("Unsupported activation function")
    def forward(self, t, z, x=None):



        if not self.homogeneous:
            if x is None:
                x = self.get_current_x(t)
            if not len(z.shape) == len(x.shape):
                x = x.unsqueeze(1)
            temp = torch.cat([z, x], dim=-1)
        else:
            temp = z

        if self.include_time:
            temp = torch.cat([t, temp], dim=-1)


        for layer in self.layers[:-1]:
            temp = self.activation(layer(temp))
        z = self.layers[-1](temp)
        return z


    def set_x(self, t, x):
        #make sure t is of shape (Batch, Time, 1)
        if len(t.shape) == 2:
            t = t.unsqueeze(-1)
        self.x = x
        self.t = t

    def get_current_x(self, t):
        """
        Retrieve the values from x corresponding to the closest indices of t.

        Args:
        t (torch.Tensor): A tensor of shape (Batch, 1) or (Batch, Time, 1) representing the time steps.

        Returns:
        torch.Tensor: A tensor of shape (Batch, Features) or (Batch, Time, Features) with the values from x.
        """
        if len(t.shape) == 2:
            # Case when t has shape (Batch, 1)
            t = t.unsqueeze(-1)  # Make t of shape (Batch, 1, 1) for consistency

        # Now t has shape (Batch, Time, 1)
        differences = torch.abs(self.t - t)
        indices = differences.argmin(dim=1)

        if indices.shape[1] == 1:
            # If t was originally (Batch, 1), we need to handle it separately
            return torch.stack([self.x[b, indices[b, 0], :] for b in range(self.x.size(0))])
        else:
            # If t was (Batch, Time, 1), return the corresponding values
            return torch.stack([self.x[b, indices[b], :] for b in range(self.x.size(0))], dim=0)


class FDE_Model(IVPRegressionModel):
    """Fractional Differential Equation model"""
    def __init__(self, config=None):
        #init super after all parameters are defined
        super(FDE_Model, self).__init__(config)
        assert config['model']['hidden_size_FDE'] > 0
        assert config['model']['n_layers_hidden_FDE'] > 0
        assert config['model']['activation_FDE'] in ['relu', 'leaky_relu', 'tanh', 'elu','sigmoid','none']
        assert config['model']['alpha'] > 0 and config['model']['alpha'] <= 1
        assert config['model']['augmentation_type'] in ['zeros', 'inital', 'none']
        if config['model']['augmentation_type'] != 'none':
            assert config['model']['augmented_dim'] > 0
        assert config['model']['include_time'] in [True, False]
        assert config['model']['homogeneous'] in [True, False]

    def define_model(self):
        dim_aug = self.config['model']['augmented_dim'] if self.config['model']['augmentation_type'] != 'none' else 0
        self.alpha=torch.Tensor([self.config['model']['alpha']]).float()



        self.FDE=FDE(self.config['data']['dim_x'],
                     self.config['data']['dim_y']+dim_aug,
                     self.config['model']['hidden_size_FDE'],
                     self.config['model']['n_layers_hidden_FDE'],
                     self.config['model']['activation_FDE'],
                     self.config['model']['include_time'],
                     self.config['model']['homogeneous'])

    def forward(self, t, x, y_0, return_z=False):

        alpha=self.alpha
        self.FDE.set_x(t,x)
        if self.config_model['augmentation_type']=='zeros':
            z_aug = torch.zeros(y_0.shape[0], self.config_model['augmented_dim'], device=x.device)
        elif self.config_model['augmentation_type']=='inital':
            #augment with the initial value of y until augmented_dim is reached
            repetitions = int(max(np.ceil(self.config_model['augmented_dim']//y_0.shape[-1]),1))
            z_aug = y_0.repeat(1,repetitions+1).reshape(y_0.shape[0],-1)
            z_aug =z_aug[:,:self.config_model['augmented_dim']]
        elif self.config_model['augmentation_type']=='none':
            z_aug = None

        if z_aug is not None:
            z_0 = torch.cat([y_0,z_aug],dim=-1)
        else:
            z_0 = y_0

        #solve fractional differential equation

        stepsize=t.diff(dim=-1).min()
        z=FDEint(self.FDE,t.unsqueeze(-1),z_0,alpha,h=stepsize,DEBUG=False)
        y = z[:, :, :self.config['data']['dim_y']]
        if return_z:
            return y,z
        return y

    def _run_epoch(self, data_loader, training=True, return_loss=False):
        """Run one epoch of training or validation.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            training (bool, optional): Whether to perform training or validation. Defaults to True.
            return_loss (bool, optional): Whether to return the loss. Defaults to False.
            return_predictions (bool, optional): Whether to return predictions. Defaults to False.

        Returns:
            float: Mean loss if return_loss is True, else None.
        """
        losses = []
        if training:
            self.train()
        else:
            self.eval()

        for batch in tqdm(data_loader, desc="Iteration Training Set" if training else "Iteration Validation Set", disable=not const.VERBOSE):
            inputs = batch['x']
            time = batch['time']
            targets = batch['y']
            y_0 = targets[:, 0, :]
            if training:
                self.optimizer.zero_grad()
            predictions = self(time, inputs, y_0)


            if training:
                loss = self.calculate_loss(predictions, targets)
                loss.backward()
                self.optimizer.step()
            if return_loss:
                loss = self.calculate_loss(predictions, targets)
                losses.append(loss.item())
        if training:
            self.scheduler.step()
        if return_loss:
            return sum(losses) / len(losses)

