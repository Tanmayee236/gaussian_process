import math
import torch
import unittest
import numpy as np
import gpytorch
import tensorflow as tf
from matplotlib import pyplot as plt
from typing import Optional
from torch import optim
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.variational import AdditiveGridInterpolationVariationalStrategy, CholeskyVariationalDistribution
from loguru import logger
from sklearn.model_selection import train_test_split
#matplotlib inline
#load_ext autoreload

train_x = torch.linspace(0, 1, 10)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.cos(train_x * (2 * math.pi))
training_iter=10



    
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
#        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# class PyTorchModel:
    
#     def __init__(self, network, loss_fn, optimizer):
#         self.network = network
#         self.loss_fn = loss_fn
#         self.optimizer = optimizer

#     def _update(self, x, y):
#         """_update Look up builder pattern.

#         This updates the network that we have (pytorch style.)

#         Args:
#             x ([type]): independent
#             y ([type]): dependent (expected)

#         Returns:
#             self: This object
#         """
#         y_pred = self.network(x)
#         loss = self.loss_fn(y_pred, y)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return self

#     def fit_one(self, x, y):
#         x = torch.FloatTensor(x)
#         y = torch.FloatTensor(y)

#         self._update(x, y)

#         return self


class KISSGPExactGPContainer:
    def __init__(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model: Optional[ExactGPModel] = None
        self.criterion: Optional[torch.nn.MSELoss] = None
        self.optimizer: Optional[torch.optim.SGD] = None
        self.index = 0
        self._max_iter = 50
    
    def _default_params(self, batch_size:int=100):
        train_x = torch.linspace(0, 1, batch_size)
        # True function is sin(2*pi*x) with Gaussian noise
        train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
        return train_x, train_y

    def train_val_dataset(self, train_x, train_y, val_split=0.25):
        X = np.array(train_x)
        y = np.array(train_y)

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        # train_idx, test_idy = train_test_split(list(range(len(_default_params.train_y))), test_size=val_split)
        return X_train, X_test, y_train, y_test


    def reset(self, batch_size:int):
        train_x, train_y = self._default_params(batch_size=batch_size)
        self.model = ExactGPModel(train_x, train_y, self.likelihood)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.15)
        self.model.train()
        self.likelihood.train()
        self._short_update(train_x, train_y)
        the_d = self.predict(train_x)
        self.index = 0
    
    def _short_update(self, X, y):
        for _ in range(int(self._max_iter/4)):
            self.index = _
            self._update_step(X, y)


    def _update_step(self, train_x, train_y):
        self.optimizer.zero_grad()
        output = self.model(train_x)
        # logger.debug(self.model.prediction_strategy)
        loss = -self.mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
            self.index + 1, training_iter, loss.item(),
            self.model.likelihood.noise.item()
        ))
        self.optimizer.step()


    def _long_update(self, X, y):
        for _ in range(self._max_iter):
            self.index = _
            self._update_step(X, y)
    
    def step(self, X, y):
        X_train, X_test, y_train, y_test = self.train_val_dataset(X, y)
        logger.debug(self.model.prediction_strategy)
        # self.model.get_fantasy_model(X_train, y_train)
        # self.model = self.model.get_fantasy_model(X_train, y_train)
        # self._long_update(X_train, y_train)
        # observed_pred, mean_abs_error = self.predict(X_test)
        # logger.warning((observed_pred, mean_abs_error))
        # return torch.linspace(0, 1, 10)

    def predict(self, test_X):
        observed_pred = self.likelihood(self.model(test_X))
        # mean_abs_error = torch.mean(torch.abs(test_y - observed_preds) / 2)
        return observed_pred
    
    
# class TestKISSGPAdditiveClassification(unittest.TestCase):

#     def __init__(self):
#         self.model: Optional[ExactGPModel] = None
#         self.criterion: Optional[torch.nn.MSELoss] = None
#         self.optimizer: Optional[torch.optim.SGD] = None
#         self.index = 0
        
        
#     def reset(self, train_x, train_y, likelihood):
#         self.model = ExactGPModel(train_x, train_y, likelihood)
#         self.mll = gpytorch.mlls.VariationalELBO(likelihood, self.model, num_data=len(train_y))
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.15)
#         self.index = 0
        
    
#     def _update(self, train_x, train_y):
#         self.optimizer.zero_grad()
#         output = self.model(train_x)
#         loss = -self.mll(output, train_y)
#         loss.backward()
#         print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
#             self.index + 1, training_iter, loss.item(),
#             self.model.likelihood.noise.item()
#         ))
#         self.optimizer.step()

#     def test_kissgp_classification_error(self):
#         # Find optimal model hyperparameters
# #        likelihood = gpytorch.likelihoods.GaussianLikelihood()
#       #  self.model = ExactGPModel(train_x, train_y, likelihood)
#         self.model.train()
#         likelihood.train()
#         self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.model)
        
#         for i in range(training_iter):
#             self.index = i
#             self._update(train_x, train_y)
        
#         test_preds = self.model(train_x).mean.ge(0.5).float()
#         mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
#         print(mean_abs_error.item())

# linrun = TestKISSGPAdditiveClassification()
# linrun.reset(train_x, train_y, likelihood)
# linrun.test_kissgp_classification_error()

# Get into evaluation (predictive posterior) mode
# model.eval()
# 
# likelihood.eval()
# Check this out!!(Tanmayee)
# https://github.com/GMvandeVen/continual-learning
# https://gpytorch.readthedocs.io/en/latest/models.html#gpytorch.models.ExactGP.get_fantasy_model
# https://gpytorch.readthedocs.io/en/latest/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html
# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     test_x = torch.linspace(0, 1, 20)
#     observed_pred = likelihood(model(test_x))
#     observed_pred



keeeeeee = KISSGPExactGPContainer()
keeeeeee.reset(batch_size=100)
_x, _y = keeeeeee._default_params()
keeeeeee.step(_x, _y)
    
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     test_x = torch.linspace(0, 1, 20)
#     observed_pred = likelihood(model(test_x))
#     # Initialize plot
#     f, ax = plt.subplots(1, 1, figsize=(4, 3))

#     # Get upper and lower confidence bounds
#     lower, upper = observed_pred.confidence_region()
#     # Plot training data as black stars
#     ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
#     # Plot predictive means as blue line
#     ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
#     # Shade between the lower and upper confidence bounds
#  #   ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     ax.set_ylim([-3, 3])
#     ax.legend(['Observed Data', 'Mean', 'Confidence'])   
#     plt.show()