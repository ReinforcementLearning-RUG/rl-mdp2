import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.mdp.mdp import MDP
from rl_mdp.mdp.reward_function import RewardFunction
from rl_mdp.mdp.transition_function import TransitionFunction
from rl_mdp.policy.policy import Policy


def create_mdp() -> AbstractMDP:
    """
    Create a simple MDP with a terminal state.
    """
    states = [0, 1, 2, 3]  # 0: S0, 1: S1, 2: S2, 3: Terminal
    actions = [0, 1]  # 0: a0, 1: a1

    transitions = {
        (0, 0): np.array([0.0, 0.8, 0.2, 0.0]),  # From state s0, a0: 80% to S1, 20% to S2
        (0, 1): np.array([0.0, 0.0, 1.0, 0.0]),  # From state s0, a1: 100% to S2
        (1, 0): np.array([0.0, 0.0, 0.5, 0.5]),  # From state s1, a0: 50% to S2, 50% to Terminal
        (1, 1): np.array([0.0, 1.0, 0.0, 0.0]),  # From state s1, a1: 100% loop in S1
        (2, 0): np.array([0.0, 0.0, 0.0, 1.0]),  # From state s2, a0: 100% to Terminal
        (2, 1): np.array([0.0, 0.0, 1.0, 0.0]),  # From state s2, a1: 100% loop in S2
        (3, 0): np.array([0.0, 0.0, 0.0, 1.0]),  # From Terminal, any action keeps you in Terminal
        (3, 1): np.array([0.0, 0.0, 0.0, 1.0]),  # From Terminal, any action keeps you in Terminal
    }

    transition_function = TransitionFunction(transitions)

    rewards = {
        (0, 0): 0.0,   # From state s0, a0: reward 0
        (0, 1): 0.0,   # From state s0, a1: reward 0
        (1, 0): 1.0,   # From state s1, a0: reward +1 (encourages moving toward the terminal state)
        (1, 1): -1.0,  # From state s1, a1: reward -1 (penalty for looping)
        (2, 0): 2.0,   # From state s2, a0: reward +2 (reward for transitioning to terminal)
        (2, 1): -1.0,  # From state s2, a1: reward -1 (penalty for looping)
        (3, 0): 0.0,   # No reward in Terminal state
        (3, 1): 0.0,   # No reward in Terminal state
    }

    reward_function = RewardFunction(rewards)

    return MDP(
        states=states,
        actions=actions,
        transition_function=transition_function,
        reward_function=reward_function,
        discount_factor=0.9,
        terminal_state=3,
        start_state=0
    )


def create_policy_1() -> Policy:
    """
    Create the first policy.
    """
    policy_1 = Policy()
    policy_1.set_action_probabilities(0, [0.2, 0.8])  # State s0: 20% to take a0, 80% to take a1
    policy_1.set_action_probabilities(1, [0.9, 0.1])  # State s1: 90% to take a0, 10% to take a1
    policy_1.set_action_probabilities(2, [1.0, 0.0])  # State s2: 100% to take a0, 0% to take a1
    policy_1.set_action_probabilities(3, [1.0, 0.0])  # Terminal state: Arbitrary action.
    return policy_1


def create_policy_2() -> Policy:
    """
    Create the second policy.
    """
    policy_2 = Policy()
    policy_2.set_action_probabilities(0, [0.7, 0.3])  # State s0: 70% to take a0, 30% to take a1
    policy_2.set_action_probabilities(1, [0.6, 0.4])  # State s1: 60% to take a0, 40% to take a1
    policy_2.set_action_probabilities(2, [0.4, 0.6])  # State s2: 40% to take a0, 60% to take a1
    policy_2.set_action_probabilities(3, [1.0, 0.0])  # Terminal state: Arbitrary action.
    return policy_2
