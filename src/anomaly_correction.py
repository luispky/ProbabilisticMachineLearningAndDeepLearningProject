import os
import numpy as np
import pandas as pd
import torch
from src.datasets import DatabaseInterface
from src.utils import cprint, bcolors, Probabilities
from copy import deepcopy


class NewInverseGradient:
    """ Author: Omar
    Bare-bones inverse gradient method
    """
    def __init__(self, model):
        self.model = model
        self._p_copy = None
        self._module = None

    def _compute_p_copy(self, p, proba, eta):
        """ Update probabilities by maintaining normalization """
        self._p_copy = p.detach().clone()
        dp = -p.grad
        self._module = np.linalg.norm(dp.numpy().flatten())
        dp = dp / self._module * eta
        self._p_copy += dp
        self._p_copy = proba.normalize(self._p_copy)

    def run(self, p: torch.tensor, structure: tuple, eta=0.01, n_iter=100, threshold_p=0.1):
        """
        Given a classifier and a set of data points, modify the data points
        so that the classification changes from 1 to 0.

        params:
        p: torch.tensor, the data point to modify
        structure: tuple, the number of values for each feature
        eta: float, the step size for the gradient descent
        n_iter: int, the maximum number of iterations
        threshold: float, the threshold probability for the loss function
        """
        assert self.model is not None
        assert p.shape[0] == 1


        assert 0 < eta < 1
        assert 0 < threshold_p < 1

        p_ = deepcopy(p)
        proba = Probabilities(structure)
        v_old = proba.onehot_to_values(p_)[0]
        success = True

        # add gaussian noise to the input
        p_.requires_grad = True

        i = 0
        while True:
            i += 1

            # Make the prediction
            y = self.model(p_)

            # Compute the loss
            loss = torch.nn.BCELoss()(y, torch.zeros_like(y))

            # Compute the gradient of the loss with respect to x
            loss.backward()

            # Create a copy of x and update the copy
            self._compute_p_copy(p_, proba, eta)

            # Update the original x with the modified copy
            p_.data = self._p_copy

            # Clear the gradient for the next iteration
            p_.grad.zero_()

            # Check if v_new is different to v_old
            v_new = proba.onehot_to_values(p_.detach().numpy())[0]
            mask = v_old != v_new
            changed = np.any(mask)

            # check if the loss is below the threshold
            loss_value = loss.item()
            print(f'\rIteration {i+1}, Loss {loss_value:.3f}', end=' ')
            if loss_value < threshold_p:
                if changed:
                    cprint(f'\rIteration {i}) loss is {loss_value:.3f} < {threshold_p}', bcolors.OKGREEN)
                    break

            # check if the gradient is zero
            if self._module == 0:
                cprint('Warning: Gradient is zero', bcolors.WARNING)
                success = False
                break

            # check if the maximum number of iterations is reached
            if i == n_iter:
                cprint('Warning: Maximum iterations reached', bcolors.WARNING)
                success = False
                break

        return {"values": v_new, "mask": mask, "proba": p_, "anomaly_p": loss_value, "success": success}


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
        self.df_x_data = df_x       # values df
        self.v_data = None          # indices
        self.p_data = None          # probabilities
        self.p_data_noisy = None    # noisy probabilities

        self.y = y

        self.p_anomaly = None       # probabilities

        # objects
        self.interface = DatabaseInterface(self.df_x_data)
        self.structure = self.interface.get_data_structure()
        self.proba = Probabilities(structure=self.structure)
        self.inv_grad = None

        # models
        self.classification_model = None
        self.diffusion_model = None

        # steps at initialization
        self._values_to_indices()
        self._indices_to_proba()
        self._compute_noisy_proba()

    def set_classification_model(self, model):
        self.classification_model = model
        self.inv_grad = NewInverseGradient(model)

    def set_diffusion_model(self, model):
        self.diffusion_model = model

    def get_value_maps(self):
        return self.interface.get_value_maps()

    def get_inverse_value_maps(self):
        return self.interface.get_inverse_value_maps()

    def _values_to_indices(self):
        """Convert the values to indices"""
        self.v_data = self.interface.convert_values_to_indices()

    def _indices_to_proba(self):
        """Convert the indices to noisy probabilities"""
        self.p_data = self.proba.to_onehot(self.v_data.to_numpy())
        del self.v_data

    def _anomaly_to_proba(self, df):
        v = self.interface.convert_values_to_indices(df)
        p = self.proba.to_onehot(v.to_numpy())
        return torch.tensor(p)

    def _compute_noisy_proba(self):
        """add noise to probabilities"""
        self.p_data_noisy = self.proba.add_noise(self.p_data)

    def get_classification_dataset(self, dtype=torch.float32):
        """Return the noisy probabilities and the anomaly labels"""
        x = self.p_data_noisy
        y = self.y.to_numpy().reshape(-1 ,1).astype(float)
        return torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype)

    def get_diffusion_dataset(self):
        """Return the dataset for the diffusion phase"""
        return self.p_data[self.y]

    def _inverse_gradient(self, p, n):
        """
        Modify p_anomaly one-by-one using the inverse gradient method
        """
        masks = []
        new_values = []
        for _ in range(n):
            results = self.inv_grad.run(p, self.structure)
            masks.append(results["mask"])
            masks.append(results["values"])
        return masks, new_values

    def _set_model(self, model):
        """Set the model to be used for anomaly correction"""
        self.model = model

    def _diffusion(self):
        """Perform diffusion on the corrected anomalies by using the masks
        self.df_x_anomaly, self.mask -> self.df_x_anomaly_corrected
        """
        v_corrected = ...

    def correct_anomaly(self, anomaly: pd.DataFrame, n):
        """Correct the anomalies in the dataset"""
        assert type(anomaly) is pd.DataFrame
        assert self.classification_model is not None, 'Please set the classification model'
        # assert self.diffusion_model is not None, 'Please set the diffusion model'


        p = self._anomaly_to_proba(anomaly)
        masks, new_values = self._inverse_gradient(p, n)
        # self._diffusion()
        return new_values


def main(data_path='..\\datasets\\sample_data_preprocessed.csv',
         model_path='..\\models\\anomaly_correction_model.pkl',
         hidden=10, loss_fn=torch.nn.BCELoss(), n_epochs=2000,
         lr=0.1, weight_decay=1e-3, momentum=0.9, nesterov=True):
    np.random.seed(42)

    # ================================================================================
    # get data
    df_x = pd.read_csv(data_path)
    df_x = df_x.sample(frac=1).reset_index(drop=True)
    df_x['Annual_Premium'] = df_x['Annual_Premium'].apply(lambda x: round(x))
    df_x['Age'] = df_x['Age'].apply(lambda x: round(x, -1))
    df_y = df_x.copy()['Response_1']
    del df_x['Response_1']

    # ================================================================================
    # anomaly_correction
    anomaly_correction = AnomalyCorrection(df_x, df_y)
    print('\nNoisy probabilities:')
    print(np.round(anomaly_correction.p_data_noisy, 2))
    print('\nValue maps:')
    for key in anomaly_correction.get_value_maps():
        print(f'{key}: {anomaly_correction.get_value_maps()[key]}')

    # ================================================================================
    # train the model
    data_p_noisy, data_y = anomaly_correction.get_classification_dataset()

    classification_model = None

    if os.path.exists(model_path):
        try:
            cprint(f'Loading model from {model_path}', bcolors.WARNING)
            classification_model = torch.load(model_path)
            cprint('Model loaded', bcolors.OKGREEN)
        except FileNotFoundError:
            cprint('Model not found', bcolors.FAIL)

    if classification_model is None:
        input_size = data_p_noisy.shape[1]
        print(f'Input size: {input_size}')

        # create the model
        cprint('Creating model', bcolors.WARNING)
        classification_model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden, 1),
            torch.nn.Sigmoid()
        )

        # optimizer
        optimizer = torch.optim.SGD(classification_model.parameters(), lr=lr,
                                    weight_decay=weight_decay, momentum=momentum,
                                    nesterov=nesterov)

        # training loop
        cprint('Training model', bcolors.WARNING)
        for epoch in range(n_epochs):
            y_pred = classification_model(data_p_noisy)
            loss = loss_fn(y_pred, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'\rlr={lr}, Epoch {epoch+1}, Loss {loss.item():.6f}', end=' ')
        print()

        # save the model
        torch.save(classification_model, model_path)
        cprint('Model saved', bcolors.OKGREEN)

    anomaly_correction.set_classification_model(classification_model)

    # ================================================================================
    # pick some anomalies
    anomaly = df_x[df_y == 1].sample(1)
    print('\nAnomaly:')
    print(anomaly)

    # ================================================================================
    # run the anomaly-correction algorithm
    corrected_anomaly = anomaly_correction.correct_anomaly(anomaly, n=5)
    print('\nCorrected anomaly:')
    print(corrected_anomaly)


if __name__ == "__main__":
    main()
