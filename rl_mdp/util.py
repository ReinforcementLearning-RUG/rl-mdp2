import numpy as np
from matplotlib import pyplot as plt
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


def evaluate_and_plot_rmse(mdp: AbstractMDP,
                           policy: AbstractPolicy,
                           true_value_function: np.ndarray,
                           alphas: np.ndarray,
                           lambdas: np.ndarray,
                           n_episodes: int) -> None:
    """
    Evaluates and plots the RMSE of state-value function estimates using Monte Carlo and TD(λ) methods.

    :param mdp: An MDP object representing the environment.
    :param policy: A policy object that provides action probabilities for each state.
    :param true_value_function: A numpy array containing the true value for each state.
    :param alphas: A numpy array of learning rates (α) to evaluate.
    :param lambdas: A numpy array of λ values for the TD(λ) algorithm.
    :param n_episodes: Number of episodes to run for each evaluation.
    """
    rmse_results = {
        'TD(λ)': {lambd: np.zeros(len(alphas)) for lambd in lambdas},
    }

    for i, alpha in enumerate(alphas):
        # TD(λ) evaluation for each λ in lambdas
        for lambd in lambdas:
            td_lambda_evaluator = TDLambdaEvaluator(policy, mdp, alpha, lambd)
            rmse_results['TD(λ)'][lambd][i] = np.sqrt(
                np.mean((true_value_function - td_lambda_evaluator.evaluate(n_episodes)) ** 2))

    plot_rmse_results(alphas, rmse_results)


def plot_rmse_results(alphas: np.ndarray, rmse_results: dict, figure_name: str = "alpha_plot") -> None:
    """
    Plots RMSE of state-value function estimates as a function of the learning rate.

    :param alphas: A numpy array of learning rates evaluated.
    :param rmse_results: A dictionary containing RMSE values for each method (MC, TD(λ)) and λ value.
    :param figure_name: Name of graph.
    """
    plt.figure(figsize=(14, 8))

    # Plot RMSE for TD(λ) for each λ value
    for lambd, rmse in rmse_results['TD(λ)'].items():
        plt.plot(alphas, rmse, label=f'TD(λ), λ={lambd}')

    plt.xlabel('Learning Rate (alpha)')
    plt.ylabel('RMSE at the End of Episode')
    plt.title('RMSE vs Learning Rate for TD(lambda) with lambda={}'.format(lambd))
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_name)
