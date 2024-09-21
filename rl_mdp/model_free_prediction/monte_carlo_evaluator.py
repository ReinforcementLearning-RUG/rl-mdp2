from collections import defaultdict
from typing import List, Tuple
import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.abstract_evaluator import AbstractEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


class MCEvaluator(AbstractEvaluator):
    def __init__(self, policy: AbstractPolicy, env: AbstractMDP, every_visit: bool = False):
        """
        Initializes the Monte Carlo Evaluator.

        :param policy: A policy object that provides action probabilities for each state.
        :param env: An environment object.
        :param first_visit: first-visit variant (True) or every-visit variant (False)
        """
        self.policy = policy
        self.env = env
        self.value_fun = np.zeros(self.env.num_states)  # State-value function approximation
        self.returns = defaultdict(list)  # Stores returns for each state
        self.every_visit = every_visit

    def evaluate(self, num_episodes: int) -> np.ndarray:
        """
        Perform the Monte Carlo prediction algorithm.
        Use the helper function _generate_episode.

        :param num_episodes: Number of episodes to run for estimating V(s).
        :return: The state-value function V(s) for the associated policy.
        """
        for _ in range(num_episodes):
            episode = self._generate_episode()
            self._update_value_function(episode)
        return self.value_fun

    def _generate_episode(self) -> List[Tuple[int, int, float]]:
        """
        Generate an episode following the policy.

        :return: A list of (state, action, reward) tuples representing the episode.
        """
        episode = []
        state = self.env.reset()
        done = False

        while not done:
            action = self.policy.sample_action(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def _update_value_function(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Update the value function using the first-visit or every-visit Monte Carlo method.

        :param episode: A list of (state, action, reward) tuples.
        """
        pass
