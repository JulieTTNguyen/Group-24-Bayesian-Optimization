import numpy as np
import sys
from pathlib import Path
import json
np.random.seed(32)
import os

project_root = Path.cwd().parent
sys.path.append(str(project_root))

from BO_functions import probability_of_improvement, expected_improvement, GP_UCB_original, squared_exponential_kernel, fit_predictive_GP, optimize_GP_hyperparams

data = np.load('game_results/game_results_humans.npz')
xyz = data['xyz']
params = data['params']
meta = data['meta']

"""WANDB integration and CLI/sweep runner"""
import argparse
import wandb
import math

wandb_key = os.getenv("WANDB_API_KEY")
if wandb_key:
    try:
        wandb.login(key=wandb_key)
    except Exception:
        pass
else:
    os.environ.setdefault("WANDB_MODE", "disabled")

"""____________________________FUNCTIONS____________________________"""

# Ground truth model in numpy
def gmm_model(params, x, y):
    # params: (N, 25), x/y: (N, 20)
    g = params[:24].reshape(6, 4)
    w, mx, my, s = g[:, 0], g[:,1], g[:, 2], g[:, 3]
    expo = -((x[...,None] - mx[None])**2 + (y[...,None] - my[None])**2) / s[None]
    res = np.sum(w[None] * np.exp(expo), axis=-1)
    
    return res * params[24]

# Softmax function for converting log probabilities to probabilities
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# frim x,y coordinates to flat index in 100x100 grid
def xy_to_flat_index(x, y):
    xi = int(np.clip(np.floor(x * 100), 0, 99))
    yi = int(np.clip(np.floor(y * 100), 0, 99))
    return yi * 100 + xi

"""_______________MAIN GP_AQUISITION EVALUATION LOOP______________"""

X = np.linspace(0,1,100,endpoint=False)
xx, yy = np.meshgrid(X,X)
Xtest = np.column_stack([xx.ravel(), yy.ravel()])

def f_objective(point,objective):
    x,y=point
    x = min(99,int(x*100))
    y = min(99, int(y*100))
    return objective[x,y]

prior_mean = 0 
prior_std = 5
betas = [0.01, 0.1, 0.3, 1, 2, 5, 10, 20, 50, 100] # for softmax temperature
xis = [0, 0.001, 0.01, 0.05, 0.1] # for EI and PI
kappas = [0.1, 0.3, 0.5, 1, 2, 5] # for UCB


def evaluate_config(beta=5.0, xi=1e-4, kappa=0.5, num_persons=20, verbose=False):
    results = []
    for i, person in enumerate(xyz[:num_persons]):
        if verbose:
            print(f"Running person {i}")
        x0 = [[x0[0],x0[1]] for x0 in person[:4]]
        objective = -gmm_model(params[i], xx, yy)
        X_sample = x0.copy()
        y_sample = [f_objective((xy[0],xy[1]),objective) for xy in x0]
        current_best = np.max(y_sample)

        for k, point in enumerate(person[4:19]):
            lengthscale, output_variance, noise_variance = optimize_GP_hyperparams(X_sample, y_sample, 500, 5e-4, prior_mean, prior_std)
            mu, covariance = fit_predictive_GP(X_sample, y_sample, Xtest, lengthscale, output_variance, noise_variance)
            std = np.sqrt(np.diag(covariance))

            acquisition_values_pi = probability_of_improvement(current_best,  mu.flatten(), std, xi)
            acquisition_values_ei = expected_improvement(current_best,  mu.flatten(), std, xi)
            acquisition_values_gp_ucb = GP_UCB_original(mu.flatten(), std, kappa)

            xh, yh = float(point[0]), float(point[1])
            idx_h = xy_to_flat_index(xh, yh)

            P_pi  = softmax(beta * acquisition_values_pi)
            P_ei  = softmax(beta * acquisition_values_ei)
            P_ucb = softmax(beta * acquisition_values_gp_ucb)

            logp_pi  = float(np.log(P_pi[idx_h]  + 1e-12))
            logp_ei  = float(np.log(P_ei[idx_h]  + 1e-12))
            logp_ucb = float(np.log(P_ucb[idx_h] + 1e-12))

            # BO-optimal points for each acquisition
            idx_PI  = int(np.argmax(acquisition_values_pi))
            idx_EI  = int(np.argmax(acquisition_values_ei))
            idx_UCB = int(np.argmax(acquisition_values_gp_ucb))

            xt_PI  = Xtest[idx_PI]
            xt_EI  = Xtest[idx_EI]
            xt_UCB = Xtest[idx_UCB]

            results.append({
                "person": int(i),
                "step": int(k),
                "beta": float(beta),
                "xi": float(xi),
                "kappa": float(kappa),
                "human_xy": [xh, yh],
                "human_z": float(point[2]),
                "idx_h": int(idx_h),
                "bo_PI_xy": xt_PI.tolist(),
                "bo_EI_xy": xt_EI.tolist(),
                "bo_UCB_xy": xt_UCB.tolist(),
                "logp_PI": logp_pi,
                "logp_EI": logp_ei,
                "logp_UCB": logp_ucb,
                "current_best": float(current_best),
            })

            X_sample.append(list(point[:2]))
            y_sample.append(point[2])
            current_best = np.max(y_sample)

    T = len(results)
    ll_EI = sum(r["logp_EI"] for r in results) / T
    ll_PI = sum(r["logp_PI"] for r in results) / T
    ll_UCB = sum(r["logp_UCB"] for r in results) / T
    mean_logp = float(np.mean([ll_EI, ll_PI, ll_UCB]))

    return {
        "ll_EI": float(ll_EI),
        "ll_PI": float(ll_PI),
        "ll_UCB": float(ll_UCB),
        "mean_logp": mean_logp,
        "T": int(T)
    }


def wandb_run():
    with wandb.init() as run:
        cfg = run.config
        metrics = evaluate_config(beta=cfg.get('beta',5.0), xi=cfg.get('xi',1e-4), kappa=cfg.get('kappa',0.5), num_persons=cfg.get('num_persons',20))
        wandb.log(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true', help='Run a wandb sweep')
    parser.add_argument('--project', type=str, default='BO_project')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--num_persons', type=int, default=20)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--xi', type=float, default=1e-4)
    parser.add_argument('--kappa', type=float, default=0.5)
    args = parser.parse_args()
    print('Running with args:', vars(args))

    if args.sweep:
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'mean_logp', 'goal': 'maximize'},
            'parameters': {
                'beta': {'distribution': 'log_uniform', 'min': 0.01, 'max': 100},
                'xi': {'distribution': 'log_uniform', 'min': 1e-6, 'max': 0.1},
                'kappa': {'distribution': 'log_uniform', 'min': 0.01, 'max': 10},
                'num_persons': {'value': args.num_persons}
            }
        }
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        wandb.agent(sweep_id, function=wandb_run)
    else:
        metrics = evaluate_config(beta=args.beta, xi=args.xi, kappa=args.kappa, num_persons=args.num_persons, verbose=True)
        print(f"Results: {metrics}")