import jax
from jax import numpy as jnp
from solver import Solver


def get_evaluator(solver: Solver, layer_index, size=None):
    in_index, out_index = solver.template.in_out_indices(layer_index)
    return jax.experimental.maps.xmap(
        lambda state, ngrad, grad, feature, pair, eps, key, temp: kfac_direct_mala(
            solver.model,
            solver.task.loss_function,
            proposer.template.layer(1 if size is None else size),
            state,
            ngrad,
            grad,
            # layer_index,
            # layer_index+1 if out_index is None else out_index,
            layer_index,
            out_index,
            feature,
            pair,
            eps,
            cfg["evo"]["steps"].get(),
            key,
            temp=temp,
        ),
        (
            [...],
            [...],
            [...],
            ["features", ...],
            ["batch", ...],
            ["features", ...],
            ["features", ...],
            [...],
        ),
        (["features", ...], ["features", ...], [...], [...]),
    )


def get_validator(layer_index, size=None):
    in_index, out_index = solver.template.in_out_indices(layer_index)
    return jax.experimental.maps.xmap(
        lambda state, ngrad, grad, feature, pair: kfac_direct_eval(
            solver.model,
            solver.task.loss_function,
            proposer.template.layer(1 if size is None else size),
            state,
            ngrad,
            grad,
            # layer_index,
            # layer_index+1 if out_index is None else out_index,
            layer_index,
            out_index,
            feature,
            pair,
        ),
        ([...], [...], [...], [...], ["batch", ...]),
        ([...], [...], [...]),
    )


def refresh_evaluators(solver: Solver, layer_only=False):
    evaluators = [get_evaluator(i) for i in range(len(solver.template.contents[:-1]))]
    validators = [get_validator(i) for i in range(len(solver.template.contents[:-1]))]
    for layer_index, con in enumerate(solver.template.contents[:-1]):
        if con is None:
            in_index, out_index = solver.template.in_out_indices(layer_index)
            new_layer_size = jnp.sum(~proposer.get_input_null(solver.state, in_index))
            evaluators[layer_index] = get_evaluator(layer_index, size=new_layer_size)
            validators[layer_index] = get_validator(layer_index, size=new_layer_size)
        elif not layer_only:
            evaluators[layer_index] = get_evaluator(layer_index)
            validators[layer_index] = get_validator(layer_index)
