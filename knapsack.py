import numpy as np

class Knapsack:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.weight = None
        self.value = None
        self.update_items()

    def update_items(self) -> None:
        self.weight = self.array[:, 0]
        self.value = self.array[:, 1]