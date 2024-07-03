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

    Forward Steps:

    values -> indices + structure
    indices -> one-hot
    one-hot -> noisy_probabilities

    ================================================================

    Sideways Steps:

    noisy_probabilities + y (anomaly labels) -> classifier
    classifier + x (anomaly) -> corrected anomaly x* (probabilities)

    ================================================================

    Backward Steps:
    x* (probabilities) -> x* (one-hot)   # diffusion!
    x* (one-hot) -> x* (indices)
    x* (indices) -> x* (values)

    ================================================================
    """
    def __init__(self, df_x: pd.DataFrame, y: pd.Series):
        self.df_x = df_x
        self.y = y

        self.interface = DatabaseInterface(self.df_x)
        self.structure = self.interface.get_data_structure()
        self.proba = Probabilities(structure=self.structure)

        self.df_x_indices = None
        self.df_x_proba = None
        self.model = None

        self._values_to_indices()
        self._indices_to_noisy_proba()

    def _values_to_indices(self):
        """Convert the values to indices"""
        self.df_x_indices = self.interface.convert_values_to_indices()

    def _indices_to_noisy_proba(self):
        """Convert the indices to noisy probabilities"""
        self.df_x_proba = self.proba.to_onehot(self.df_x_indices.to_numpy())
        del self.df_x_indices
        self.df_x_proba = self.proba.add_noise(self.df_x_proba)

    def _inverse_gradient(self, anomalies):
        v_corrected = ...
        masks = ...
        return v_corrected, masks

    def _get_noisy_proba_data(self):
        """Return the noisy probabilities and the anomaly labels"""
        return self.df_x_proba, self.y

    def _set_model(self, model):
        """Set the model to be used for anomaly correction"""
        self.model = model

    def _diffusion(self, v_corrected, mask):
        """Perform diffusion on the corrected anomalies"""
        v_corrected = ...
        return v_corrected

    def correct_anomaly(self, anomalies):
        """Correct the anomalies in the dataset"""
        v_corrected, masks = self._inverse_gradient(anomalies)
        v_corrected = self._diffusion(v_corrected, masks)
        return v_corrected


def main(data_path='..\\datasets\\sample_data_preprocessed.csv',
         model_path='..\\models\\anomaly_correction_model.pkl'):

    # get data
    df_x = pd.read_csv(data_path)
    df_x = df_x.sample(frac=1).reset_index(drop=True)
    df_x['Annual_Premium'] = df_x['Annual_Premium'].apply(lambda x: round(x))
    df_x['Age'] = df_x['Age'].apply(lambda x: round(x, -1))
    df_y = df_x.copy()['Response_1']
    del df_x['Response_1']

    # train model
    anomaly_correction = AnomalyCorrection(df_x, df_y)

    print(np.round(anomaly_correction.df_x_proba, 2))



if __name__ == "__main__":
    main()
