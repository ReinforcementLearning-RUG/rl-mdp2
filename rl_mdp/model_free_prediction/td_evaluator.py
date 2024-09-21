import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.abstract_evaluator import AbstractEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


class TDEvaluator(AbstractEvaluator):
    def __init__(self, policy: AbstractPolicy, env: AbstractMDP, alpha: float = 0.1, n: int = 1):
        """
        Initializes the n-step TD Evaluator.

        :param policy: A policy object that provides action probabilities for each state.
        :param env: A mdp object.
        :param alpha: The step size (learning rate).
        """
        self.policy = policy
        self.env = env
        self.alpha = alpha
        self.n = n
        self.value_fun = np.zeros(self.env.num_states)

    def evaluate(self, num_episodes: int) -> np.ndarray:
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
        Runs a single episode using the n-step TD method to update the value function.
        """
        state_buffer = np.zeros(self.n + 1, dtype=int)
        reward_buffer = np.zeros(self.n + 1)
        terminal_time = float('inf')  # Time when the episode ends.
        time_step = 0  # Current time step `t`.
        # Use name 'update time' for 'tau' from pseudocode.

        state_buffer[0] = self.env.reset()  # Initialize first state

        while True:
            if time_step < terminal_time:
                action = self.policy.sample_action(int(state_buffer[time_step % (self.n + 1)]))
                next_state, reward, done = self.env.step(action)
                state_buffer[(time_step + 1) % (self.n + 1)] = next_state
                reward_buffer[(time_step + 1) % (self.n + 1)] = reward
                if done:
                    terminal_time = time_step + 1
            # Complete or add helper function.
