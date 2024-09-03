import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from abc import ABC, abstractmethod
import json
from pathlib import Path
import Constants as const
from tqdm import tqdm
import logging
from src.utils.helpers import ensure_sequential_dataloader, concatenate_batches

class LoggingMixin:
    """A mixin class that adds logging functionality."""
    def __init__(self, *args, **kwargs):
        super(LoggingMixin, self).__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log(self, message):
        self.logger.info(message)

class TimestampMixin:
    """A mixin class that adds timestamp functionality."""
    def timestamp(self):
        import datetime
        return datetime.datetime.now()

class DummyScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class BaseRegressionModel(nn.Module, ABC, LoggingMixin, TimestampMixin):
    """Base class for all regression models

    This class defines a template for regression models, handling common tasks such as
    training, validation, prediction, and saving/loading configurations.
    """
    def __init__(self, config=None):
        super(BaseRegressionModel, self).__init__()
        LoggingMixin.__init__(self)
        self.config_train = config['training'] if config else {}
        self.config_model = config['model'] if config else {}
        self.config = config
        self.define_model()
        self.optimizer = self._initialize_optimizer()
        if config['training']['gamma_scheduler'] != None:
            self.scheduler=torch.optim.lr_scheduler.ExponentialLR(self.optimizer,config['training']['gamma_scheduler'])
        else:
            self.scheduler = DummyScheduler(self.optimizer)

    @abstractmethod
    def define_model(self):
        """Define the model architecture. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def forward(self, t, x, return_z=False):
        """Forward pass of the model. Must be implemented by subclasses.

        Args:
            t (torch.Tensor): Time tensor.
            x (torch.Tensor): Input tensor.
            return_z (bool, optional): Whether to return latent variables. Defaults to False.

        Returns:
            torch.Tensor: Model predictions.
        """
        pass

    @staticmethod
    def loss_fn(*args, **kwargs):
        return F.mse_loss(*args, **kwargs)

    def trainable_parameters(self):
        """Get the list of trainable parameters.

        Returns:
            list: List of trainable parameters.
        """
        return list(self.parameters())

    def calculate_loss(self, predictions, targets):
        """Calculate the loss between predictions and targets.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.

        Returns:
            torch.Tensor: Calculated loss.
        """
        return self.loss_fn(predictions, targets)

    def _initialize_optimizer(self):
        """Initialize the optimizer for the model.

        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        """
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        return Adam(self.trainable_parameters(), lr=lr)

    def _run_epoch(self, data_loader, training=True, return_loss=False):
        """Run one epoch of training or validation.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            training (bool, optional): Whether to perform training or validation. Defaults to True.
            return_loss (bool, optional): Whether to return the loss. Defaults to False.

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
            if training:
                self.optimizer.zero_grad()
            predictions = self(time, inputs)
            targets = targets.to(dtype=predictions.dtype)
            loss = self.calculate_loss(predictions, targets)
            if training:
                loss.backward()
                self.optimizer.step()
            if return_loss:
                losses.append(loss.item())
        if training:
            self.scheduler.step()

        if return_loss:
            return sum(losses) / len(losses)

    def train_step(self, data_loader, return_loss=False):
        """Perform a training step.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            return_loss (bool, optional): Whether to return the loss. Defaults to False.

        Returns:
            float: Mean loss if return_loss is True, else None.
        """
        return self._run_epoch(data_loader, training=True, return_loss=return_loss)

    def validate_step(self, data_loader):
        """Perform a validation step.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

        Returns:
            float: Mean loss.
        """
        return self._run_epoch(data_loader, training=False, return_loss=True)

    def predict(self, data_loader, samples_only=False, return_raw=False):
        """Predict on the given data_loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            samples_only (bool, optional): Whether to return raw predictions and targets. Defaults to False.

        Returns:
            tuple: Predictions, ground truth, inputs, time steps, and latent variables (if applicable).
        """
        inputs = []
        predictions = []
        truth = []
        ts = []
        z = []
        data_loader = ensure_sequential_dataloader(data_loader)
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Prediction Set", disable=not const.VERBOSE):
                x = batch['x']
                time = batch['time']
                targets = batch['y']
                y_0 = targets[:, 0, :]
                y_hat, z_batch = self(time, x,y_0, return_z=True)
                inputs.append(x)
                predictions.append(y_hat)
                truth.append(targets)
                ts.append(time)
                z.append(z_batch)
            if samples_only:
                if return_raw:
                    return predictions, truth, inputs, ts, z
                predictions=torch.cat(predictions,dim=0)
                truth=torch.cat(truth,dim=0)
                inputs=torch.cat(inputs,dim=0)
                ts=torch.cat(ts,dim=0)
                z=torch.cat(z,dim=0)
                return predictions, truth, inputs, ts, z
            predictions = concatenate_batches(predictions)
            truth = concatenate_batches(truth)
            inputs = concatenate_batches(inputs)
            ts = concatenate_batches(ts)
            z = concatenate_batches(z)
            return predictions, truth, inputs, ts, z

    def save_model(self, path):
        """Save the model to the given path.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load the model from the given path.

        Args:
            path (str): Path to load the model from.
        """
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode

    def save_config(self, path, config=None):
        """Save the configuration to the given path.

        Args:
            config (dict): Configuration dictionary.
            path (str): Path to save the configuration.
        """
        if config is None:
            config = self.config
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def load_config(path):
        """Load the configuration from the given path.

        Args:
            path (str): Path to load the configuration from.

        Returns:
            dict: Loaded configuration dictionary.
        """
        with open(path, 'r') as f:
            config = json.load(f)
        return config

    @staticmethod
    def initialize_from_config(config_path, initialize_model=False):
        """Load the configuration from the given path and optionally initialize the model.

        Args:
            config_path (str): Path to load the configuration from.
            initialize_model (bool, optional): Whether to initialize the model. Defaults to False.

        Returns:
            dict or BaseRegressionModel: Configuration dictionary or initialized model.
        """
        config = BaseRegressionModel.load_config(config_path)
        if initialize_model:
            model_class = globals()[config['model']['class_name']]
            model = model_class(config)
            return model
        return config

    def save_model_and_config(self, path):
        """Save the model and its configuration to the given path.

        Args:
            path (str or Path): Directory path to save the model and configuration.
        """
        path = Path(path)
        # Ensure the directory exists
        path.mkdir(parents=True, exist_ok=True)
        # Save the model state dictionary
        model_path = path / 'model.pth'
        torch.save(self.state_dict(), model_path)
        # Save the configuration
        config_path = path / 'config.json'
        self.save_config(config_path)

    @staticmethod
    def load_model_and_config(path):
        """Load the model and its configuration from the given path.

        Args:
            path (str or Path): Directory path to load the model and configuration.

        Returns:
            BaseRegressionModel: The loaded model with its configuration.
        """
        path = Path(path)
        # Load the configuration
        config_path = path / 'config.json'
        config = BaseRegressionModel.load_config(config_path)
        # Initialize the model
        model_class = globals()[config['model']['class_name']]
        model = model_class(config)
        # Load the model state dictionary
        model_path = path / 'model.pth'
        model.load_model(model_path)
        return model

    @staticmethod
    def initialize_from_config(config_path, initialize_model=False, model_path=None):
        """Load the configuration from the given path and optionally initialize the model.

        Args:
            config_path (str or Path): Path to load the configuration from.
            initialize_model (bool, optional): Whether to initialize the model. Defaults to False.
            model_path (str or Path, optional): Path to the saved model directory. Defaults to None.

        Returns:
            dict or BaseRegressionModel: Configuration dictionary or initialized model.
        """
        config_path = Path(config_path)
        config = BaseRegressionModel.load_config(config_path)
        if initialize_model:
            model_class = globals()[config['model']['class_name']]
            model = model_class(config)
            if model_path:
                model_path = Path(model_path)
                model.load_model(model_path)
            return model
        return config

    def save_config(self, path, config=None):
        """Save the configuration to the given path.

        Args:
            config (dict, optional): Configuration dictionary.
            path (str or Path): Path to save the configuration.
        """
        if config is None:
            config = self.config
        path = Path(path)
        with path.open('w') as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def load_config(path):
        """Load the configuration from the given path.

        Args:
            path (str or Path): Path to load the configuration from.

        Returns:
            dict: Loaded configuration dictionary.
        """
        path = Path(path)
        with path.open('r') as f:
            config = json.load(f)
        return config

    def load_model(self, path):
        """Load the model from the given path.

        Args:
            path (str or Path): Path to load the model from.
        """
        path = Path(path)
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode


class IVPRegressionModel(BaseRegressionModel):
    """Base class for Inital Value Problem IVP regression models"""

    def __init__(self, config=None):
        super(IVPRegressionModel, self).__init__(config)

    def forward(self, t, x, y_0=None, return_z=False):
        """Forward pass of the NODE model.

        Args:
            t (torch.Tensor): Time tensor.
            x (torch.Tensor): Input tensor.
            y_0 (torch.Tensor, optional): Initial condition. Defaults to None.
            return_z (bool, optional): Whether to return latent variables. Defaults to False.

        Returns:
            torch.Tensor: Model predictions.
        """
        pass  # Implement the forward pass specific to NODE

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

            loss = self.calculate_loss(predictions, targets)
            if training:
                loss.backward()
                self.optimizer.step()
            if return_loss:
                losses.append(loss.item())
        if training:
            self.scheduler.step()
        if return_loss:
            return sum(losses) / len(losses)



    def train_step(self, data_loader, return_loss=False):
        """Perform a training step.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            return_loss (bool, optional): Whether to return the loss. Defaults to False.

        Returns:
            float: Mean loss if return_loss is True, else None.
        """
        return self._run_epoch(data_loader, training=True, return_loss=return_loss)

    def validate_step(self, data_loader,return_predictions=False):
        """Perform a validation step.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

        Returns:
            float: Mean loss.
        """
        return self._run_epoch(data_loader, training=False, return_loss=True)


class IVPRegressionModelLatEncoder(BaseRegressionModel):
    """Base class for NODE regression models with latent encoder"""

    def __init__(self, config=None):
        super(IVPRegressionModelLatEncoder, self).__init__(config)

    def forward(self, t, x, z_0=None, return_z=False):
        """Forward pass of the NODE model with latent encoder.

        Args:
            t (torch.Tensor): Time tensor.
            x (torch.Tensor): Input tensor.
            z_0 (torch.Tensor, optional): Initial latent condition. Defaults to None.
            return_z (bool, optional): Whether to return latent variables. Defaults to False.

        Returns:
            torch.Tensor: Model predictions.
        """
        pass  # Implement the forward pass specific to NODE with latent encoder

    def train_step(self, data_loader, return_loss=False):
        """Perform a training step.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            return_loss (bool, optional): Whether to return the loss. Defaults to False.

        Returns:
            float: Mean loss if return_loss is True, else None.
        """
        return self._run_epoch(data_loader, training=True, return_loss=return_loss)

    def validate_step(self, data_loader):
        """Perform a validation step.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

        Returns:
            float: Mean loss.
        """
        return self._run_epoch(data_loader, training=False, return_loss=True)

    def predict(self, data_loader, samples_only=False, return_z=False):
        """Predict on the given data_loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            samples_only (bool, optional): Whether to return raw predictions and targets. Defaults to False.
            return_z (bool, optional): Whether to return latent variables. Defaults to False.

        Returns:
            tuple: Predictions, ground truth, inputs, time steps, and latent variables (if applicable).
        """
        inputs = []
        predictions = []
        truth = []
        ts = []
        z_hat = []
        z_true = []
        data_loader = ensure_sequential_dataloader(data_loader)
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Prediction Set", disable=not const.VERBOSE):
                x = batch['x']
                time = batch['time']
                targets = batch['y']
                latent = batch['z']
                z_out = targets[:, 0, :]
                if self.encode_latent_space:
                    enc_input = torch.cat([inputs[:, 0:self.len_encode, :], targets[:, 0:self.len_encode, :]], dim=-1)
                    z_lat = self.encoder(enc_input)
                else:
                    z_lat = latent[:, 0, :]
                z_0 = torch.cat([z_out, z_lat], dim=-1)
                y_hat, z_hat_batch = self(time, x, z_0, return_z=True)
                inputs.append(x)
                predictions.append(y_hat)
                truth.append(targets)
                ts.append(time.unsqueeze(2))
                z_hat.append(z_hat_batch)
                z_true.append(latent)
            if samples_only:
                if return_z:
                    return predictions, truth, inputs, ts, z_hat, z_true
                return predictions, truth, inputs, ts
            predictions = concatenate_batches(predictions)
            truth = concatenate_batches(truth)
            inputs = concatenate_batches(inputs)
            ts = concatenate_batches(ts)
            z_hat = concatenate_batches(z_hat)
            z_true = concatenate_batches(z_true)
            if return_z:
                return predictions, truth, inputs, ts, z_hat, z_true

            return predictions, truth, inputs, ts
