import torch
import pandas as pd
from src.inverse_gradient import InverseGradient
from src.datasets import DatabaseInterface
from src.utils import Probabilities

# set default type to avoid problems with gradient
DEFAULT_TYPE = torch.float64
torch.set_default_dtype(DEFAULT_TYPE)


class AnomalyCorrection:
    """
    Take as input a dataset, train a model, and
    perform anomaly correction.

    ================================================================

    Steps at Initialization:

    values -> indices + structure -> one-hot -> noisy_probabilities

    ================================================================

    Training:

    noisy_probabilities + y (anomaly labels) -> classifier
    genuine datapoint -> diffusion model

    ================================================================

    Steps at Correction:

    x_anomaly -> p_anomaly (onehot)
    Inverse gradient: classifier + p_anomaly -> corrected p_anomaly*
    Diffusion: p_anomaly* -> p_anomaly**
    p_anomaly** (probabilities) -> p_anomaly** (one-hot) -> v_anomaly** (indices) x_anomaly** (values)

    ================================================================
    """
    def __init__(self, df_x: pd.DataFrame, y: pd.Series, noise=0.):
        self.df_x_data = df_x       # values df
        self.y = y
        self.noise = noise
        self.v_data = None          # indices
        self.p_data = None          # probabilities
        self.p_data_noisy = None    # noisy probabilities
        self.p_anomaly = None       # probabilities

        # objects
        self.interface = DatabaseInterface(self.df_x_data)
        self.structure = self.interface.get_data_structure()
        self.proba = Probabilities(structure=self.structure)
        self.inv_grad = None

        # model
        self.classification_model = None
        self.diffusion = None

        # steps at initialization
        self._values_to_indices()
        self._indices_to_proba()
        self._compute_noisy_proba()

    def set_classification_model(self, model):
        self.classification_model = model
        self.inv_grad = InverseGradient(model)

    def set_diffusion(self, diffusion):
        """set diffusion model"""
        self.diffusion = diffusion

    def get_value_maps(self):
        """get value maps"""
        return self.interface.get_value_maps()

    def get_inverse_value_maps(self):
        """get inverse value maps"""
        return self.interface.get_inverse_value_maps()

    def _values_to_indices(self):
        """Convert the values to indices"""
        self.v_data = self.interface.convert_values_to_indices()

    def _indices_to_proba(self):
        """Convert the indices to noisy probabilities"""
        self.p_data = self.proba.to_onehot(self.v_data.to_numpy())

    def _anomaly_to_proba(self, df, dtype=DEFAULT_TYPE):
        """convert anomaly values to probabilities"""
        self.anomaly_indices = self.interface.convert_values_to_indices(df).to_numpy()
        self.anomaly_p = self.proba.to_onehot(self.anomaly_indices)
        self.anomaly_p = torch.tensor(self.anomaly_p, dtype=dtype)
        return self.anomaly_p

    def _compute_noisy_proba(self):
        """add noise to probabilities"""
        self.p_data_noisy = self.proba.add_noise(self.p_data)

    def get_classification_dataset(self, dtype=DEFAULT_TYPE):
        """Return the noisy probabilities and the anomaly labels"""
        x = self.p_data_noisy
        y = self.y.to_numpy().reshape(-1, 1).astype(float)
        return torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype)

    def get_diffusion_dataset(self):
        """Return the dataset for the diffusion phase
        returns: Dataset without anomalies in index space
        """
        return self.v_data[~self.y]

    def _inverse_gradient(self, p, n):
        """ Modify p_anomaly one-by-one using the inverse gradient method """

        masks = []
        new_values = []
        for _ in range(n):
            p_ = self.proba.add_noise(p, k=self.noise)
            results = self.inv_grad.run(p_, self.structure)
            masks.append(results["mask"])
            new_values.append(results["values"])
        return masks, new_values

    def correct_anomaly(self, anomaly: pd.DataFrame, n):
        """Correct the anomalies in the dataset"""
        assert type(anomaly) is pd.DataFrame
        assert self.classification_model is not None, 'Please set the classification model'
        # assert self.diffusion is not None, 'Please set the diffusion model'

        p = self._anomaly_to_proba(anomaly)
        masks, new_indices = self._inverse_gradient(p, n)

        print('\nanomaly_indices')
        print(self.anomaly_indices)

        print('\nmasks')
        for mask in masks:
            print(f'{mask}  ({len(mask)})')
        print(len(masks))

        print('\nstructure')
        print(self.proba.structure)

        print('\nindices before diffusion')

        new_indices = self.diffusion.inpaint(anomaly_indices=self.anomaly_indices, masks=masks, proba=self.proba)

        print(new_indices.shape)

        print('\nindices after diffusion')
        new_values = self.interface.convert_indices_to_values(new_indices)
        print('\nvalues after diffusion')

        return new_values
