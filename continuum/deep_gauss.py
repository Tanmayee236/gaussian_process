import os
import urllib.request
from math import floor
import gpytorch
import math
import gpytorch
import torch
import tqdm
from loguru import logger
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import AddedLossTerm, DeepApproximateMLL, VariationalELBO
from gpytorch.models import GP, ApproximateGP
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  VariationalStrategy)
from scipy.io import loadmat
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset
from continuum.hidden import DeepGPHiddenLayer
from continuum.loaders import ElevatorLoader


# # this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)

# if not smoke_test and not os.path.isfile('../elevators.mat'):
#     print('Downloading \'elevators\' UCI dataset...')
#     urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')


# if smoke_test:  # this is for running the notebook in our testing framework
#     X, y = torch.randn(1000, 3), torch.randn(1000)
# else:
#     data = torch.Tensor(loadmat('../elevators.mat')['data'])
#     X = data[:, :-1]
#     X = X - X.min(0)[0]
#     X = 2 * (X / X.max(0)[0]) - 1
#     y = data[:, -1]


# train_n = int(floor(0.8 * len(X)))
# train_x = X[:train_n, :].contiguous()
# train_y = y[:train_n].contiguous()

# test_x = X[train_n:, :].contiguous()
# test_y = y[train_n:].contiguous()

# if torch.cuda.is_available():
#     train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()


# train_dataset = TensorDataset(train_x, train_y)
# train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

elevator_loader = ElevatorLoader()
train_loader, test_loader, x_shape = elevator_loader.get_data()


num_output_dims = 2 if smoke_test else 10


class DeepGPSystem(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
        )

        last_layer = DeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = model.likelihood(model(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

model = DeepGPSystem(x_shape)
if torch.cuda.is_available():
    model = model.cuda()

num_epochs = 1 if smoke_test else 10
num_samples = 3 if smoke_test else 10


optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, x_shape[-2]))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            minibatch_iter.set_postfix(loss=loss.item())


# test_dataset = TensorDataset(test_x, test_y)
# test_loader = DataLoader(test_dataset, batch_size=1024)

model.eval()
predictive_means, predictive_variances, test_lls = model.predict(test_loader)

logger.info((predictive_means, predictive_variances, predictive_variances))

# rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
# logger.warning(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")