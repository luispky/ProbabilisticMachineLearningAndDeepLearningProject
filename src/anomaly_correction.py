import os
import numpy as np
import pandas as pd
import torch
from src.datasets import DatabaseInterface
from src.utils import cprint, bcolors, Probabilities
from src.inverse_gradient import InverseGradient


DEFAULT_MODEL_NAME = f'model_{os.path.splitext(os.path.basename(__file__))[0]}.pkl'


class AnomalyCorrection:
    """
    Take as input a dataset, train a model, and
    perform anomaly correction.

    ================================================================

    Steps at Initialization:

    values -> indices + structure
    indices -> one-hot
    one-hot -> noisy_probabilities

    ================================================================

    Training:

    noisy_probabilities + y (anomaly labels) -> classifier
    genuine datapoint -> diffusion model

    ================================================================

    Steps at Correction:

    x_anomaly -> p_anomaly (onehot)
    Inverse gradient: classifier + p_anomaly -> corrected p_anomaly*
    Diffusion: p_anomaly* -> p_anomaly**
    p_anomaly** (probabilities) -> p_anomaly** (one-hot)
    p_anomaly** (one-hot) -> v_anomaly** (indices)
    v_anomaly** (indices) -> x_anomaly** (values)

    ================================================================
    """
    def __init__(self, df_x: pd.DataFrame, y: pd.Series):
        self.df_x_data = df_x   # values df
        self.v_data = None      # indices
        self.p_data = None      # probabilities
        self.l_data = None      # logits

        self.y = y

        self.df_x_anomaly = None    # values
        self.v_anomaly = None       # indices
        self.p_anomaly = None       # probabilities
        self.l_anomaly = None       # logits

        self.interface = DatabaseInterface(self.df_x_data)
        self.structure = self.interface.get_data_structure()
        self.proba = Probabilities(structure=self.structure)

        # models
        self.classification_model = None
        self.diffusion_model = None

        self._values_to_indices()
        self._indices_to_noisy_proba()

    def get_value_maps(self):
        return self.interface.get_value_maps()

    def get_inverse_value_maps(self):
        return self.interface.get_inverse_value_maps()

    def _values_to_indices(self):
        """Convert the values to indices"""
        self.v_data = self.interface.convert_values_to_indices()

    def _indices_to_noisy_proba(self):
        """Convert the indices to noisy probabilities"""
        self.p_data = self.proba.to_onehot(self.v_data.to_numpy())
        del self.v_data
        self.p_data = self.proba.add_noise(self.p_data)

    def get_probability_dataset(self):
        """Return the noisy probabilities and the anomaly labels"""
        return self.p_data, self.y

    def _inverse_gradient(self):
        """modify p_anomaly one-by-one using the inverse gradient method"""
        v_corrected = ...
        masks = ...

    def _get_noisy_proba_data(self):
        """Return the noisy probabilities and the anomaly labels"""
        return self.p_data, self.y

    def _set_model(self, model):
        """Set the model to be used for anomaly correction"""
        self.model = model

    def _diffusion(self):
        """Perform diffusion on the corrected anomalies by using the masks
        self.df_x_anomaly, self.mask -> self.df_x_anomaly_corrected
        """
        v_corrected = ...

    def correct_anomaly(self, anomalies):
        """Correct the anomalies in the dataset"""
        assert type(anomalies) is pd.DataFrame
        # assert self.classification_model is not None
        # assert self.diffusion_model is not None

        self.df_x_anomaly = anomalies
        # self._inverse_gradient()
        # self._diffusion()
        return self.df_x_anomaly


def main(data_path='..\\datasets\\sample_data_preprocessed.csv',
         model_path='..\\models\\anomaly_correction_model.pkl',
         n_examples=5):
    np.random.seed(42)

    # get data
    df_x = pd.read_csv(data_path)
    df_x = df_x.sample(frac=1).reset_index(drop=True)
    df_x['Annual_Premium'] = df_x['Annual_Premium'].apply(lambda x: round(x))
    df_x['Age'] = df_x['Age'].apply(lambda x: round(x, -1))
    df_y = df_x.copy()['Response_1']
    del df_x['Response_1']

    # train model
    anomaly_correction = AnomalyCorrection(df_x, df_y)
    print('\nNoisy probabilities:')
    print(np.round(anomaly_correction.p_data, 2))
    print('\nValue maps:')
    for key in anomaly_correction.get_value_maps():
        print(f'{key}: {anomaly_correction.get_value_maps()[key]}')

    # pick some anomalies
    anomalies = df_x[df_y == 1].sample(n_examples)
    print('\nAnomalies:')
    print(anomalies)

    # run the anomaly-correction algorithm
    corrected_anomalies = anomaly_correction.correct_anomaly(anomalies)
    print('\nCorrected anomalies:')
    print(corrected_anomalies)


if __name__ == "__main__":
    main()
