import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.utils import Probabilities


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
            self.dataset = self._generate_data()
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
    def __init__(self, size, n_values: tuple, threshold: float):
        super().__init__(size)
        self.n_values = n_values
        self.threshold = threshold
        self.label_values = None

    def _generate_data(self):
        """
        Generate a dataset in probability space that represents arrays of label encoded categories.
        The y labels are binary, True/Anomaly if the sum of the values in the array exceeds the threshold.
        """
        proba = Probabilities(self.n_values)

        # raw data
        p = np.random.random(size=(self.size, sum(self.n_values)))
        p = proba.normalize(p)

        x = proba.prob_to_onehot(p)
        self.label_values = proba.onehot_to_values(x)
        y = np.sum(self.label_values, axis=1) > self.threshold
        y = np.expand_dims(y, axis=1)

        # convert to torch tensors
        x = torch.tensor(p, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.bool)

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
        U, S, V = torch.svd(cov_tensor)
        transform_matrix = U @ torch.diag(torch.sqrt(S))

        normal_samples = torch.randn(self.size, len(self.mean))
        samples = normal_samples @ transform_matrix.T + mean_tensor

        return samples
