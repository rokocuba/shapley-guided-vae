from __future__ import annotations

import torch

from .game import ReconstructionGame


class MonteCarloShapleyEstimator:
    def __init__(self, game: ReconstructionGame, n_permutations: int = 64) -> None:
        self.game = game
        self.n_permutations = n_permutations

    def estimate(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement permutation sampling for online Shapley approximation.
        raise NotImplementedError
