import os
import torch
import pandas as pd
import numpy as np
from src.datasets import DatabaseInterface
from utils import cprint, bcolors, Probabilities
from copy import deepcopy

from src.denoising_diffusion_pm import DDPMAnomalyCorrection as Diffusion
from utils import plot_loss


# set default type to avoid problems with gradient
DEFAULT_TYPE = torch.float32
torch.set_default_dtype(DEFAULT_TYPE)


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
        if self._module == 0:
            raise ZeroDivisionError
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
        v_new = v_old.copy()
        mask = v_new == v_new
        success = True

        # add gaussian noise to the input
        p_.requires_grad = True

        i = 0
        while True:
            i += 1

            # Make the prediction
            y = self.model(p_)
            p_anomaly = y[0][0]

            # Compute the loss
            loss = torch.nn.BCELoss()(y, torch.zeros_like(y))

            # Compute the gradient of the loss with respect to x
            loss.backward()

            # Create a copy of x and update the copy
            try:
                self._compute_p_copy(p_, proba, eta)
            except ZeroDivisionError:
                # check if the gradient is zero
                cprint('Warning: Gradient is zero', bcolors.WARNING)
                success = False
                break

            # Update the original x with the modified copy
            p_.data = self._p_copy

            # Clear the gradient for the next iteration
            p_.grad.zero_()

            # Check if v_new is different to v_old
            v_new = proba.onehot_to_values(p_.detach().numpy())[0]
            mask = v_old != v_new
            changed = np.any(mask)

            # check if the loss is below the threshold
            print(f'\rIteration {i+1}, pred {p_anomaly:.1%}', end=' ')
            if p_anomaly < threshold_p:
                if changed:
                    cprint(f'\rIteration {i}) loss is {p_anomaly:.1%} < {threshold_p:.1%}', bcolors.OKGREEN)
                    break

            # check if the maximum number of iterations is reached
            if i == n_iter:
                cprint(f'Warning: Maximum iterations reached ({p_anomaly:.1%})', bcolors.WARNING)
                success = False
                break

        return {"values": v_new, "mask": mask, "proba": p_, "anomaly_p": p_anomaly, "success": success}


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

        # steps at initialization
        self._values_to_indices()
        self._indices_to_proba()
        self._compute_noisy_proba()

    def set_classification_model(self, model):
        self.classification_model = model
        self.inv_grad = NewInverseGradient(model)

    def set_diffusion(self, diffusion):
        self.diffusion = diffusion

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

    def _anomaly_to_proba(self, df, dtype=DEFAULT_TYPE):
        self.anomaly_indices = self.interface.convert_values_to_indices(df).to_numpy()
        self.anomaly_p = self.proba.to_onehot(self.anomaly_indices)
        self.anomaly_p = torch.tensor(self.anomaly_p , dtype=dtype)
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
        """
        Modify p_anomaly one-by-one using the inverse gradient method
        """
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

        print('\nindices after diffusion')
        new_values = self.interface.convert_indices_to_values(new_indices)
        print('\nvalues after diffusion')

        return new_values


class ClassificationModel:
    """
    Example classifier
    """
    def __init__(self):
        self.model = None

    def load_from_file(self, model_path):
        """
        Set the model if the pkl file is found.
        If a file is not found, then the model remains None
        """
        if os.path.exists(model_path):
            try:
                cprint(f'Loading model from {model_path}', bcolors.WARNING)
                self.model = torch.load(model_path)
                cprint('Model loaded', bcolors.OKGREEN)
            except FileNotFoundError:
                cprint('Model not found', bcolors.FAIL)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def reset(self, input_size, hidden):
        """ Create the model """
        print(f'Input size: {input_size}')
        cprint('Creating model', bcolors.WARNING)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden, 1),
            torch.nn.Sigmoid()
        )

    def _training_loop(self, num_samples, optimizer, n_epochs, batch_size, x, y, loss_fn):

        for epoch in range(n_epochs):
            total_loss = 0.0
            for i in range(0, num_samples, batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y[i:i + batch_size]

                y_pred = self.model(batch_x)
                loss = loss_fn(y_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (num_samples / batch_size)
            print(f'\rEpoch {epoch + 1}, Loss {avg_loss:.6f}', end=' ')
        print()

    def train(self, x, y, model_path, loss_fn, n_epochs=200, lr=0.1, weight_decay=1e-4,
              momentum=0.9, nesterov=True, batch_size=100):
        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,
                                    weight_decay=weight_decay, momentum=momentum,
                                    nesterov=nesterov)

        # Training loop
        num_samples = x.shape[0]
        self._training_loop(num_samples, optimizer, n_epochs, batch_size, x, y, loss_fn)

        # test the model
        y_pred = self.model(x)
        loss = loss_fn(y_pred, y)
        print(f'Final Loss {loss.item():.6f}')

        # performance metrics
        y_class = (y_pred > 0.5).float()
        accuracy = np.array(y_class == y).astype(float).mean()
        dummy_acc = max(y.mean().item(), 1 - y.mean().item())
        acc = accuracy.item()
        usefulness = max([0, (acc - dummy_acc) / (1 - dummy_acc)])
        if usefulness > 0.75:
            color = bcolors.OKGREEN
        elif usefulness > 0.25:
            color = bcolors.WARNING
        else:
            color = bcolors.FAIL
        print(f'Dummy accuracy = {dummy_acc:.1%}')
        print(f'Accuracy on test data = {acc:.1%}')
        cprint(f'usefulness = {usefulness:.1%}', color)
        rmse = torch.sqrt(torch.mean((y_pred - y) ** 2))
        print(f'RMSE on test data {rmse.item():.3f}')

        # save the model
        torch.save(self.model, model_path)
        cprint('Model saved', bcolors.OKGREEN)


def main(data_path='..\\datasets\\sum_limit_problem.csv',
         model_path='..\\models\\anomaly_correction_model.pkl',
         hidden=10, loss_fn=torch.nn.MSELoss(), n_epochs=250):
    np.random.seed(42)

    # ================================================================================
    # get data
    df_x = pd.read_csv(data_path)
    df_x = df_x.sample(frac=1).reset_index(drop=True)
    df_y = df_x.copy()['anomaly']
    del df_x['anomaly']

    # ================================================================================
    # anomaly_correction
    anomaly_correction = AnomalyCorrection(df_x, df_y, noise=1.)
    print('\nNoisy probabilities:')
    print(np.round(anomaly_correction.p_data_noisy, 2))
    print('\nValue maps:')
    for key in anomaly_correction.get_value_maps():
        print(f'{key}: {anomaly_correction.get_value_maps()[key]}')

    # ================================================================================
    # The classification model
    data_x, data_y = anomaly_correction.get_classification_dataset()

    classification_model = ClassificationModel()
    classification_model.load_from_file(model_path)

    if classification_model.model is None:
        classification_model.reset(input_size=data_x.shape[1], hidden=hidden)
        classification_model.train(data_x, data_y,
                                   model_path, loss_fn, n_epochs=n_epochs)

    anomaly_correction.set_classification_model(classification_model)

    # ================================================================================
    # The diffusion model
    # self.ddpm_scheduler = None
    data_diff_x = anomaly_correction.get_diffusion_dataset()
    dataset_shape = data_diff_x.shape
    
    # in index space
    print(f'\ndata_diff_x {data_diff_x.shape}')
    print(data_diff_x)

    # This class takes as input the anomaly and the masks, and returns the modified anomalies
    diffusion = Diffusion(dataset_shape=dataset_shape, 
                          noise_time_steps=128)
    
    ddpm_model_name = 'ddpm_model'
    
    diffusion.load_model_pickle(ddpm_model_name) #!name, NOT PATH

    if diffusion.model is None:
        diffusion.set_model(time_dim_emb=64,
                            concat_x_and_t=True,
                            feed_forward_kernel=True,
                            hidden_units=[2*dataset_shape[1],
                                          3*dataset_shape[1], 
                                          2*dataset_shape[1] 
                                          ],
                            unet=False)  
        # DDPM training
        train_losses = diffusion.train(data_diff_x, 
                                       batch_size=16,
                                       learning_rate=1e-3,
                                       epochs=100,
                                       beta_ema=0.999,
                                       plot_data=True,
                                       structure=anomaly_correction.structure,
                                       original_data_name='ddpm_original_data')
        loss_name = 'ddpm_loss'
        
        plot_loss(train_losses, loss_name, save_locally=True)
        diffusion.save_model_pickle(filename=ddpm_model_name, 
                                    ema_model=True)

    print(f'\ndiffusion.model')
    print(diffusion.model)

    anomaly_correction.set_diffusion(diffusion)

    # ================================================================================
    # pick some anomalies
    anomaly = df_x[df_y == 1].sample(1)
    print('\nAnomaly:')
    print(anomaly)

    # ================================================================================
    # run the anomaly-correction algorithm
    corrected_anomaly = anomaly_correction.correct_anomaly(anomaly, n=10)
    print('\nCorrected anomaly:')
    for v_ in corrected_anomaly:
        print(v_)


if __name__ == "__main__":
    main()
