import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import IVPRegressionModel
from torchdiffeq import odeint as odeint
import numpy as np



class ODE(nn.Module):
    """A simple n-layer feed forward neural network"""

    def __init__(self, dim_x, dim_z, hidden_size, n_layers_hidden=3, activation='relu', include_time=True,Homogeneous=False):
        super().__init__()
        self.include_time = include_time
        self.homogeneous = Homogeneous
        self.layers = nn.ModuleList()
        if include_time and not Homogeneous:
            self.layers.append(nn.Linear(1+dim_x+dim_z, hidden_size))
        elif include_time and Homogeneous:
            self.layers.append(nn.Linear(1+dim_z, hidden_size))
        elif not include_time and  Homogeneous:
            self.layers.append(nn.Linear(dim_z, hidden_size))
        elif not include_time and not Homogeneous:
            self.layers.append(nn.Linear(dim_x+dim_z, hidden_size))

        for _ in range(n_layers_hidden - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, dim_z))

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation =='elu':
            self.activation = F.elu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise ValueError("Unsupported activation function")

    def set_x(self, t, x):
        # make sure t is of shape (Batch, Time, 1)
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

    def forward(self, t, z, x=None):

        if len(t.shape)<2:
            if len(t.shape)==0:
                t = t.unsqueeze(0).unsqueeze(1)
            t = t.repeat(z.shape[0], 1)

        if not self.homogeneous:
            if x is None:
                x = self.get_current_x(t)
            temp=torch.cat([z, x], dim=-1)
        else:
            temp=z
        if self.include_time:
            temp = torch.cat([t, temp], dim=-1)

        for layer in self.layers[:-1]:
            temp = self.activation(layer(temp))
        z = self.layers[-1](temp)
        return z


class NODE_Model(IVPRegressionModel):
    def __init__(self, config):
        super(NODE_Model, self).__init__(config)

        assert config['model']['hidden_size_ODE'] > 0
        assert config['model']['n_layers_hidden_ODE'] > 0
        assert config['model']['activation_ODE'] in ['relu', 'leaky_relu', 'tanh', 'elu','sigmoid','none']

        assert config['model']['augmentation_type'] in ['zeros',
                                                       'inital',
                                                       'none'], "Augmentation type must be 'zeros' or 'inital'"
        if config['model']['augmentation_type'] is not 'none':
            assert config['model']['augmented_dim'] > 0, "Augmented dimension must be greater than 0"
        assert config['model']['include_time'] in [True, False], "Include time must be True or False"
        assert config['model']['homogeneous'] in [True, False], "Homogeneous must be True or False"
        assert config['model']['solver'] in ['rk4', 'dopri5', 'explicit_adams'], "Solver must be 'rk4', 'dopri5' or 'adams'"


    def define_model(self):
        config_model=self.config['model']
        dim_x = self.config['data']['dim_x']
        dim_aug = self.config['model']['augmented_dim'] if self.config['model']['augmentation_type']!='none' else 0
        dim_z = self.config['data']['dim_y']+dim_aug
        hidden_size = config_model['hidden_size_ODE']
        n_layers_hidden = config_model['n_layers_hidden_ODE']
        activation = config_model['activation_ODE']
        include_time = config_model['include_time']
        homogeneous = config_model['homogeneous']


        self.ODE = ODE(dim_x, dim_z, hidden_size, n_layers_hidden, activation, include_time,homogeneous)


    def forward(self, t, x, y_0, return_z=False):
        self.ODE.set_x(t,x)

        if self.config_model['augmentation_type']=='zeros':
            z_aug = torch.zeros(x.shape[0], self.config_model['augmented_dim'], device=x.device)
        elif self.config_model['augmentation_type']=='inital':
            #augment with the initial value of y until augmented_dim is reached
            repetitions = int(max(np.ceil(self.config_model['augmented_dim']//y_0.shape[-1]),1))
            z_aug = y_0.repeat(1,repetitions+1).reshape(x.shape[0],-1)
            z_aug =z_aug[:,:self.config_model['augmented_dim']]
        elif self.config_model['augmentation_type']=='none':
            z_aug = None



        if z_aug is not None:
            z_0 = torch.cat([y_0,z_aug],dim=-1)
        else:
            z_0 = y_0
        t = t[0].squeeze(0) #odeint expects t to be 1D (as we expect all samples to be equally sampled we take the first sample)
        z = odeint(self.ODE, z_0, t, method='rk4').permute(1, 0, 2)
        y=z[:,:,:self.config['data']['dim_y']]

        if return_z:
            return y,z
        return y