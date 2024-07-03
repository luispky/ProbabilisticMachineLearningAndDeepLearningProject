import os
import numpy as np
import torch
from src.utils import cprint, bcolors, Probabilities
from src.inverse_gradient import InverseGradient
from src.datasets import SumCategoricalDataset


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
    def __init__(self):
        ...
