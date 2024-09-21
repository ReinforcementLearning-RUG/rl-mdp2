from abc import ABC, abstractmethod
from typing import Dict


class AbstractEvaluator(ABC):

    @abstractmethod
    def evaluate(self, num_episodes: int) -> Dict[int, float]:
        pass
