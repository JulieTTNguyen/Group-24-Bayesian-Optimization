import numpy as np
import torch
from torch import nn, distributions
from scipy.spatial.distance import cdist
from scipy.stats import norm
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.FloatTensor)
def probability_of_improvement(current_best, mean, std, xi):

    # since std can be 0, to avoid an error, we add a small value in the denominator (like +1e-9)
    PI =  norm.cdf((mean - current_best - xi) / (std + 1e-9))
    return PI 


def expected_improvement(current_best, mean, std, xi):
    # start by computing the Z as we did in the probability of improvement function
    # to avoid division by 0, add a small term eg. np.spacing(1e6) to the denominator
    Z = (mean - current_best - xi) / (std + 1e-9) # or Z = (mean - current_best - eps) / (std + np.spacing(1e6))
    # now we have to compute the output only for the terms that have their std > 0
    EI = (mean - current_best - xi) * norm.cdf(Z) + std * norm.pdf(Z)
    EI[std == 0] = 0
    
    return EI


def GP_UCB_original(mean, std, kappa):
       return mean + kappa * std


def squared_exponential_kernel(x, y, lengthscale, variance):
    '''
    Function that computes the covariance matrix using a squared-exponential kernel
    '''
    # pair-wise distances, size: NxM
    sqdist = cdist(x,y, 'sqeuclidean')
    # compute the kernel
    cov_matrix = variance * np.exp(-0.5 * sqdist * (1/lengthscale**2))  # NxM
    return cov_matrix


def fit_predictive_GP(X, y, Xtest, lengthscale, kernel_variance, noise_variance):

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    Xtest = np.asarray(Xtest)
    K = squared_exponential_kernel(X, X, lengthscale, kernel_variance)
    L = np.linalg.cholesky(K + noise_variance * np.eye(len(X)))

    # compute the mean at our test points.
    Ks = squared_exponential_kernel(X, Xtest, lengthscale, kernel_variance)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))  #
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    # compute the variance at our test points.
    Kss = squared_exponential_kernel(Xtest, Xtest, lengthscale, kernel_variance)
    covariance = Kss - (v.T @ v)
    
    return mu, covariance


# I am using PyTorch to define the optimization function, it can be done in different other ways.
# It is not the best way to implement it, I suppose
def optimize_GP_hyperparams(Xtrain, ytrain, optimization_steps, learning_rate, mean_prior, prior_std):  
    # we are re-defining the kernel because we need it in PyTorch
    def squared_exponential_kernel_torch(x, y, lengthscale, variance):
        x_sq = torch.sum(x**2, dim=1, keepdim=True)
        y_sq = torch.sum(y**2, dim=1, keepdim=True).T
        sqdist = x_sq + y_sq - 2 * x @ y.T
        return variance * torch.exp(-0.5 * sqdist / lengthscale**2)

    X = np.array(Xtrain)
    y = np.array(ytrain).reshape(-1,1)
    N = len(X)

    # tranform our training set in Tensor
    Xtrain_tensor = torch.from_numpy(X).float()
    ytrain_tensor = torch.from_numpy(y ).float()
    # we should define our hyperparameters as torch parameters where we keep track of
    # the operations to get hte gradients from them
    _lambda = nn.Parameter(torch.tensor(1.), requires_grad=True)
    output_variance = nn.Parameter(torch.tensor(1.), requires_grad=True)
    noise_variance = nn.Parameter(torch.tensor(.5), requires_grad=True)

    # we use Adam as optimizer
    optim = torch.optim.Adam([_lambda, output_variance, noise_variance], lr=learning_rate)

    # optimization loop using the log-likelihood that involves the cholesky decomposition 
    nlls = []
    lambdas = []
    output_variances = []
    noise_variances = []
    iterations = optimization_steps
    for i in range(iterations):
        assert noise_variance.item() >= 0, f"ouch! {i, noise_variance}"
        optim.zero_grad()

        jitter = 1e-6

        K = squared_exponential_kernel_torch(Xtrain_tensor, Xtrain_tensor, _lambda,
                                                output_variance) + (noise_variance+jitter) * torch.eye(N)
        L = torch.linalg.cholesky(K)

        _alpha_temp = torch.linalg.solve_triangular(L, ytrain_tensor,upper=False)
        _alpha = torch.linalg.solve_triangular(L.t(),_alpha_temp,upper=True)
        nll = N / 2 * torch.log(torch.tensor(2 * np.pi)) + 0.5 * torch.matmul(ytrain_tensor.transpose(0, 1),
                                                                              _alpha) + torch.sum(torch.log(torch.diag(L)))

        # we have to add the log-likelihood of the prior
        norm = distributions.Normal(loc=mean_prior, scale=prior_std)
        prior_negloglike =  torch.log(_lambda) - norm.log_prob(_lambda)

        nll += 0.9 * prior_negloglike
        nll.backward()

        nlls.append(nll.item())
        lambdas.append(_lambda.item())
        output_variances.append(output_variance.item())
        noise_variances.append(noise_variance.item())
        optim.step()

        # ---- constraints (after optimizer update) ----
        _lambda.data.clamp_(min=1e-6)
        output_variance.data.clamp_(min=1e-6)
        noise_variance.data.clamp_(min=1e-5, max=0.05)
        
    return _lambda.item(), output_variance.item(), noise_variance.item()