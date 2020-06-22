import torch
import gpytorch
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class GeneralApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(GeneralApproximateGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MeanFieldApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = (
            gpytorch.variational
            .MeanFieldVariationalDistribution(
                inducing_points.size(-2)
            )
        )
        variational_strategy = (
            gpytorch
            .variational
            .VariationalStrategy(
                self, inducing_points,
                variational_distribution,
                learn_inducing_locations=True
            )
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
