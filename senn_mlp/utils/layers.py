from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import lax
from senn_utils.solver import Solver


@dataclass
class WidthModification:
    solver: Solver
    dataset: Any
    labels: Any
    ratio: float
    layer_index: int
    new_state: Any

    def apply(self):
        assert self.solver.template.contents[self.layer_index] is not None
        prev_loss = self.solver.train_batch(
            (self.dataset, self.labels), observe_only=True, loud=True
        )
        self.solver.state = self.new_state
        new_size = self.solver.template.contents[self.layer_index] + 1
        self.solver.template.contents[self.layer_index] = new_size
        refresh_evaluators(layer_only=True)
        print(f"new size for layer {self.layer_index}: {new_size}")
        new_loss = self.solver.train_batch(
            (self.dataset, self.labels), observe_only=True, loud=True
        )
        assert new_loss / prev_loss < 1.001, "adding width made loss worse"


@dataclass
class DepthModification:
    solver: Solver
    dataset: Any
    labels: Any
    ratio: float
    layer_index: int
    new_state: Any

    def apply(self):
        # raise NotImplementedError
        assert self.solver.template.contents[self.layer_index] is None
        prev_loss = self.solver.train_batch(
            (self.dataset, self.labels), observe_only=True, loud=True
        )
        old_state = self.solver.state
        self.solver.state = self.new_state
        assert (
            self.layer_index > 0
        ), "cannot guess new size because there is no preceding layer"
        new_size = self.solver.template.contents[self.layer_index - 1]
        self.solver.template.contents[self.layer_index] = new_size
        refresh_evaluators()
        print(f"Created new layer at {self.layer_index} with size: {new_size}")
        new_loss = self.solver.train_batch(
            (self.dataset, self.labels), observe_only=True, loud=True
        )
        if new_loss / prev_loss >= 1.2:
            print(f"old state: {old_state}")
            print(f"new state: {self.new_state}")
            assert new_loss / prev_loss < 1.2, "adding layer made loss worse"


def layer_residual(A, res, act):
    Ainv = jnp.linalg.pinv(A)
    Cra = lax.pmean(res[:, None] * act[None, :], "batch")
    lres = res - Cra @ Ainv @ act
    return lres
