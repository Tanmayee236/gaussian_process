import os
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP

from continuum.models.gauss.deep import hidden


smoke_test = ('CI' in os.environ)
num_output_dims = 2 if smoke_test else 10


class BaseDeepGPSystem(DeepGP):
    def __init__(self, train_x_shape: torch.Tensor):
        hidden_layer = hidden.ApproximateDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
        )

        last_layer = hidden.ApproximateDeepGPHiddenLayer(
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

    # def predict(self, test_loader):
    #     with torch.no_grad():
    #         mus = []
    #         variances = []
    #         lls = []
    #         for x_batch, y_batch in test_loader:
    #             preds = model.likelihood(model(x_batch))
    #             mus.append(preds.mean)
    #             variances.append(preds.variance)
    #             lls.append(
    #                 (
    #                     model
    #                     .likelihood
    #                     .log_marginal(y_batch, model(x_batch))
    #                 )
    #             )

    #     return (
    #             torch.cat(mus, dim=-1),
    #             torch.cat(variances, dim=-1),
    #             torch.cat(lls, dim=-1)
    #         )
