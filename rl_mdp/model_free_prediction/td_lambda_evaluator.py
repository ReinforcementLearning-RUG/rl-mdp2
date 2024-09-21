from collections import defaultdict
from typing import Dict

from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.abstract_evaluator import AbstractEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


class TDLambdaEvaluator(AbstractEvaluator):
    def __init__(self,
                 policy: AbstractPolicy,
                 env: AbstractMDP,
                 alpha: float = 0.1,
                 lambd: float = 0.9):
        """
        Initializes the TD(λ) Learner.

        :param policy: A policy object that provides action probabilities for each state.
        :param env: A mdp object.
        :param alpha: The step size (learning rate).
        :param lambd: The trace decay parameter (λ).
        """
        super().__init__(policy, env, alpha)
        self.value_fun = defaultdict(float)  # State-value function approximation.
        self.lambd = lambd
        self.eligibility_traces = defaultdict(float)

    def evaluate(self, num_episodes: int) -> Dict[int, float]:
        """
        Perform the TD prediction algorithm.

        :param num_episodes: Number of episodes to run for estimating V(s).
        :return: The state-value function V(s) for the associated policy.
        """
        for _ in range(num_episodes):
            self._run_episode()

        return self.value_fun

    def _run_episode(self) -> None:
        """
        Runs a single episode using the TD(λ) method to update the value function.
        """
        pass
