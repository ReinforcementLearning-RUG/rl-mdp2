from collections import defaultdict
from typing import List, Tuple, Dict
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.policy.abstract_policy import AbstractPolicy


class TDEvaluator:
    def __init__(self, policy: AbstractPolicy, env: AbstractMDP, alpha: float = 0.1):
        """
        Initializes the TD(0) Learner.

        :param policy: A policy object that provides action probabilities for each state.
        :param env: An environment object following the OpenAI Gym API.
        :param alpha: The step size (learning rate).
        """
        self.policy = policy
        self.env = env
        self.alpha = alpha
        self.value_fun = defaultdict(float)  # State-value function approximation

    def evaluate(self, num_episodes: int) -> Dict[int, float]:
        """
        Perform the TD(0) prediction algorithm.

        :param num_episodes: Number of episodes to run for estimating V(s).
        :return: The state-value function V(s) for the associated policy.
        """
        pass


