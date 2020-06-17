   
import math
import torch
import gpytorch
import tensorflow as tf
from matplotlib import pyplot as plt

from torch import optim
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.variational import AdditiveGridInterpolationVariationalStrategy, CholeskyVariationalDistribution

train_x = torch.linspace(0, 1, 10)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
training_iter=100
likelihood = gpytorch.likelihoods.GaussianLikelihood()
observed_pred=0

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

    
class TestKISSGPAdditiveClassification():

    def __init__(self):
        self.model: Optional[GPClassificationModel] = None
        self.criterion: Optional[torch.nn.MSELoss] = None
        self.optimizer: Optional[torch.optim.SGD] = None

        
        
    def reset(self, train_x, train_y, likelihood):
        self.model = ExactGPModel(train_x, train_y, likelihood)
        self.mll = gpytorch.mlls.VariationalELBO(likelihood, self.model, num_data=len(train_y))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.15)
        
        

        
    def test_kissgp_classification_error(self):
        self.reset(train_x, train_y, likelihood)
        self.model.train()
        likelihood.train()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.model)
        
        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.model.likelihood.noise.item()
            ))
            self.optimizer.step()
        
        test_preds = self.model(train_x).mean.ge(0.5).float()
        mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
        print(mean_abs_error.item())
        
    def evaluate(self):
        self.model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 2, 20)
            observed_pred = likelihood(self.model(test_x))
#            print(observed_pred.loc)
        return observed_pred, test_x
            
    def plot(self):   
        with torch.no_grad():
            preds, test_x = linrun.evaluate()
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))
            lower, upper = preds.confidence_region()
            # Plot training data as black stars
            ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), preds.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])

linrun = TestKISSGPAdditiveClassification()
linrun.reset(train_x, train_y, likelihood)
linrun.test_kissgp_classification_error()
a =linrun.evaluate()
#test_x = torch.linspace(0, 1, 20)
linrun.plot()   
plt.show()    
