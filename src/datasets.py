import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from src.utils import Probabilities


class DatabaseInterface:
    """
    This class is used to preprocess an actual dataset
    to something that can be used by our algorithm.
    """
    NUMBER_VALUES_LIMIT = 100

    def __init__(self, df: pd.DataFrame):
        self.original_df = df
        self.value_maps = None
        self.inverse_value_maps = None
        self._init_value_map()

    def _init_value_map(self):
        """Compute the value maps for each column in the dataframe"""
        self.value_maps = dict()
        for col in self.original_df.columns:
            unique_values = self.original_df[col].unique()
            unique_values = list(unique_values)
            unique_values.sort(key=lambda x: str(x))
            value_map = {k: v for v, k in enumerate(unique_values)}
            if len(value_map) > self.NUMBER_VALUES_LIMIT:
                raise ValueError(f'Too many unique values in column {col} '
                                 f'({len(value_map)} > {self.NUMBER_VALUES_LIMIT})')
            self.value_maps[col] = value_map
        self.inverse_value_maps = {col: {v: k for k, v in self.value_maps[col].items()} for col in self.value_maps}

    def convert_values_to_indices(self, df: pd.DataFrame=None):  # todo check
        """
        Convert the original dataframe to a dataframe of indices.
        """
        if df is None:
            df = self.original_df.copy()
        for col in df.columns:
            df[col] = df[col].map(self.value_maps[col])
        return df

    def convert_indices_to_values(self, df: pd.DataFrame):
        """
        Convert a dataframe of indices to a dataframe of values.
        """
        df = df.copy()
        for col in df.columns:
            df[col] = df[col].map(self.inverse_value_maps[col])
        return df

    def get_value_maps(self):
        return self.value_maps

    def get_inverse_value_maps(self):
        return self.inverse_value_maps

    def get_data_structure(self):
        return tuple([len(self.value_maps[col]) for col in self.value_maps])


class BaseDataset:
    def __init__(self, size):
        self.dataset = None
        self.size = size

    def _generate_data(self):
        """
        Assign the dataset to self.dataset, a dataset dictionary with keys 'x' and 'y'
        """
        raise NotImplementedError

    def get_data(self) -> dict:
        """
        Anomaly if sum of numbers is more than k
        """
        if self.dataset is None:
            self._generate_data()
        assert type(self.dataset) is dict, 'self.dataset must be a dictionary, please modify ._generate_data()'
        return self.dataset

    def get_dataloader(self, batch_size, shuffle=True, with_labels=False):
        """
        Generate a dataloader for the dataset.
        """
        data_x = self.get_data()['x']

        if with_labels:
            data_y = self.get_data()['y']
            tensor_dataset = TensorDataset(data_x, data_y)
        else:
            tensor_dataset = TensorDataset(data_x)

        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_dataset_shape(self):
        """ Return the shape of the dataset for the x values. """
        assert self.dataset is not None, 'Dataset not generated'
        return self.dataset['x'].shape


class SumCategoricalDataset(BaseDataset):
    """ Author Omar
    Anomaly if sum of numbers is more than threshold
    """
    def __init__(self, size, structure: tuple, threshold: float):
        super().__init__(size)
        self.structure = structure
        self.threshold = threshold
        self.label_values = None

    def _generate_data(self):
        """
        Generate a dataset in probability space that represents arrays of label encoded categories.
        The y labels are binary, True/Anomaly if the sum of the values in the array exceeds the threshold.
        """
        proba = Probabilities(self.structure)

        # raw data
        p = np.random.random(size=(self.size, sum(self.structure)))
        p = proba.normalize(p)

        x = proba.prob_to_onehot(p)
        self.label_values = proba.onehot_to_values(x)
        y = np.sum(self.label_values, axis=1) > self.threshold
        y = np.expand_dims(y, axis=1)

        # convert to torch tensors
        x = torch.tensor(p, dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64)

        self.dataset = {'x': x, 'y': y}


class GaussianDataset(BaseDataset):
    """
    Author: Luis
    Class to generate the dataset for the DDPM model.
    """
    def __init__(self, size, mean, cov):
        super().__init__(size)
        self.mean = mean
        self.cov = cov

    def _generate_data(self):
        """
        Generates samples using an alternative approach to handle non-positive definite covariance matrices.
        """
        mean_tensor = torch.tensor(self.mean, dtype=torch.float32)
        cov_tensor = torch.tensor(self.cov, dtype=torch.float32)

        # Ensure the covariance matrix is symmetric
        cov_tensor = (cov_tensor + cov_tensor.T) / 2

        # Use SVD to generate samples
        u, s, v = torch.svd(cov_tensor)
        transform_matrix = u @ torch.diag(torch.sqrt(s))

        normal_samples = torch.randn(self.size, len(self.mean))
        samples = normal_samples @ transform_matrix.T + mean_tensor

        return samples
