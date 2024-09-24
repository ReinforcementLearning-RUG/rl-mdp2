from abc import ABC, abstractmethod
from typing import Dict
from rl_mdp.policy.abstract_policy import AbstractPolicy


class AbstractEvaluator(ABC):

    @abstractmethod
    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> Dict[int, float]:
        pass
