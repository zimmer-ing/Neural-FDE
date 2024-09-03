
from abc import abstractmethod
from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from pathlib import Path
import numpy as np

class DatasetBase(Dataset):
    """Abstract class to create a dataset from a directory of files using the TimeSeriesBase class"""

    def __init__(self, directory,file_extension=".csv", config=None,device=None):
        if config is None:
            config = {}
        self.directory = directory
        self.file_extension = file_extension
        self.config = config
        self.device = device or 'cpu'


        self.timeseries_datasets = self.load_dataset()



    def to(self, device):
        """
        Move all sub-datasets to the specified device.
        """
        for ts_dataset in self.timeseries_datasets:
            ts_dataset.to(device)

    def __len__(self):
        return len(self.timeseries_datasets)


    def __getitem__(self, index):
        #get the index of the dataset

        return self.timeseries_datasets[index]

    def create_subsequence(self, data, channels,device='cpu'):
        """Creates subsequence from the data"""


        #get names of channels
        time_channel, x_channels, y_channels,z_channels = channels
        #get data from channels
        time_data = data[time_channel].values
        x_data = data[x_channels].values
        y_data = data[y_channels].values
        if len(z_channels)>0:
            z_data = data[z_channels].values
        else:
            z_data = np.zeros_like(y_data)

        time_subsequence = torch.tensor(time_data, dtype=torch.float32, device=device)
        x_subsequence = torch.tensor(x_data, dtype=torch.float32, device=device)
        y_subsequence = torch.tensor(y_data, dtype=torch.float32, device=device)
        z_subsequence = torch.tensor(z_data, dtype=torch.float32, device=device)
        return time_subsequence, x_subsequence, y_subsequence, z_subsequence

    def load_dataset(self):
        """Loads the dataset from the directory"""
        channels=self.get_channels()
        directory_path = Path(self.directory)
        all_files = list(directory_path.glob(f'*{self.file_extension}'))
        dataset = []

        for file_path in all_files:
            data= self.load_file(file_path)
            dataset.append(self.create_subsequence(data, channels,device=self.device))
        return dataset

    @staticmethod
    def collate_fn(batch):
        """Collates batch of data."""
        time, x, y, z = zip(*batch)
        return {"time": torch.stack(time),
                "x": torch.stack(x),
                "y": torch.stack(y),
                "z": torch.stack(z)}


    @abstractmethod
    def load_file(self, file_path) -> pd.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def get_channels() -> Tuple[str, List[str], List[str], List[str]]:
        """Returns the channel names for the time, X, Y ,Z channels"""
        pass
