from functools import partial
from typing import Any

import jax
from jax import lax
from jax import numpy as jnp
from jax.tree_util import tree_map


def kfac_observe(model, loss_function, state, tangent, pair):
    # y, dL, Jt, aux, grad, res
    datum, label = pair
    y, Jt, aux = jax.jvp(
        fun=lambda state: model.apply(state, datum, mutable="intermediates"),
        primals=(state,),
        tangents=(tangent,),
        has_aux=True,
    )
    dloss = jax.grad(partial(loss_function, label))(y)
    loss_sqnorm = lax.pmean(jnp.sum(dloss**2), "batch")

    _, backward = jax.vjp(
        lambda state: model.apply(state, datum),
        state,
    )
    (grad,) = backward(dloss / jnp.sqrt(jnp.sum(dloss**2)))
    Jt_rescale = lax.psum(jnp.sum(Jt * dloss), "batch") / (
        lax.psum(jnp.sum(Jt * Jt), "batch") + 1e-10
    )
    lres = dloss - Jt * Jt_rescale
    (state_residual,) = backward(lres)

    acts = model.apply(aux, method=model.extract_activations)
    out_grads = model.apply(grad, method=model.extract_out_grads)
    out_ress = model.apply(state_residual, method=model.extract_out_grads)

    As = [lax.pmean(a[:, None] * a[None, :], "batch") for (a,) in acts]
    Gs = [lax.pmean(g[:, None] * g[None, :], "batch") for g in out_grads]

    return As, Gs, acts, out_ress, out_grads, loss_sqnorm


def kfac_single_eval(fmodel, act, Ginv, res, feature):
    fval = fmodel.apply(feature, act)
    A = lax.pmean(fval[..., :, None] * fval[..., None, :], "batch")
    Ainv = jnp.linalg.pinv(A)
    corr = lax.pmean(fval[:, None] * res[None, :], "batch")
    normed_corr = Ainv @ corr @ Ginv
    rscore = jnp.sum(corr * normed_corr)
    return rscore


def kfac_single_sgd(fmodel, act, Ginv, res, feature, eps, atikh=0.0):
    fval, backward = jax.vjp(lambda theta: fmodel.apply(theta, act), feature)
    A = lax.pmean(fval[..., :, None] * fval[..., None, :], "batch")
    Ainv = jnp.linalg.pinv(tikhonov(A, atikh))
    corr = lax.pmean(fval[:, None] * res[None, :], "batch")

    normed_corr = Ainv @ corr @ Ginv
    rscore = jnp.sum(corr * normed_corr)

    resres = res - corr.T @ Ainv @ fval
    fgrad = normed_corr @ resres
    (theta_grad,) = tree_map(lambda arr: lax.pmean(arr, "batch"), backward(fgrad))

    def ldet(feat):
        L = feat["params"]["linear"]["kernel"]
        sign, logabs = jnp.linalg.slogdet(L.T @ L)
        return logabs

    REGULARISE = 1e-2
    lnmag, lnmag_grad = jax.value_and_grad(ldet)(feature)

    new_feature = tree_map(
        lambda theta, g, lgrad: theta + eps * g - REGULARISE * lnmag * lgrad,
        feature,
        theta_grad,
        lnmag_grad,
    )
    return rscore, new_feature


def kfac_mala_burst(
    fmodel,
    act,
    Ginv,
    res,
    feature,
    key,
    lr,
    temp=1e0,
    atikh=0.0,
    steps=10,
    score_norm=1e0,
):
    priorvar = get_prior_var(feature)

    def score_cotan(fval):
        A = lax.pmean(fval[..., :, None] * fval[..., None, :], "batch")
        Ainv = jnp.linalg.pinv(tikhonov(A, atikh))
        corr = lax.pmean(fval[:, None] * res[None, :], "batch") / jnp.sqrt(score_norm)

        normed_corr = Ainv @ corr @ Ginv
        rscore = jnp.sum(corr * normed_corr)

        resres = res - corr.T @ Ainv @ fval
        fgrad = normed_corr @ resres
        return rscore, fgrad

    def loss_grad(feature):
        fval, backward = jax.vjp(lambda theta: fmodel.apply(theta, act), feature)
        rscore, cotangent = score_cotan(fval)
        (theta_grad,) = tree_map(
            lambda arr: lax.pmean(arr, "batch"), backward(cotangent)
        )
        return -rscore, theta_grad

    feature, accept_rate = mala_steps(
        loss_grad, priorvar, feature, key, lr, steps, temp=temp
    )
    rscore, _ = score_cotan(fmodel.apply(feature, act))
    return rscore * score_norm, feature, accept_rate


def kfac_direct_mala(
    model,
    loss_function,
    fmodel,
    state,
    tangent,
    full_grad,
    in_index,
    out_index,
    feature,
    pair,
    lr,
    steps,
    key,
    temp=1e0,
):
    As, Gs, acts, resids, grads, loss_sqnorm = kfac_observe(
        model, loss_function, state, tangent, pair
    )
    A, G, (act_in,), (act_out,) = (
        As[out_index],
        Gs[out_index],
        acts[in_index],
        acts[out_index],
    )
    res, grad = resids[out_index], grads[out_index]
    Ginv = jnp.linalg.pinv(tikhonov(G, 1e-1 * meandiag(G)))
    atikh = 0e-1 * meandiag(A)
    res = layer_residual(A, res, act_out)
    lin_grad = full_grad["params"][f"layers_{out_index}"]["linear"]["kernel"]
    layer_score = layer_baseline(A, G, lin_grad)

    @flax.struct.dataclass
    class State:
        lr: float
        rscore: float
        feature: Any

    init_state = State(lr, 0.0, feature)

    def fun(state, key):
        new_rscore, new_feature, accept_rate = kfac_mala_burst(
            fmodel,
            act_in,
            Ginv,
            res,
            state.feature,
            key,
            state.lr,
            temp=temp,
            score_norm=layer_score,
        )
        TARGET = 0.6
        changed_lr = jnp.where(accept_rate > TARGET, state.lr * 1.3, state.lr / 1.3)
        new_lr = jnp.where(jnp.abs(accept_rate - TARGET) > 0.3, changed_lr, state.lr)
        return State(new_lr, new_rscore, new_feature), accept_rate

    final_state, accepts = lax.scan(fun, init_state, jax.random.split(key, steps))
    return final_state.rscore, final_state.feature, layer_score, loss_sqnorm


def kfac_direct_eval(
    model,
    loss_function,
    fmodel,
    state,
    tangent,
    full_grad,
    in_index,
    out_index,
    feature,
    pair,
):
    As, Gs, acts, resids, _, loss_sqnorm = kfac_observe(
        model, loss_function, state, tangent, pair
    )
    A, G, (act_in,), (act_out,), res = (
        As[out_index],
        Gs[out_index],
        acts[in_index],
        acts[out_index],
        resids[out_index],
    )
    Ginv = jnp.linalg.pinv(G)
    res = layer_residual(A, res, act_out)
    rscore = kfac_single_eval(fmodel, act_in, Ginv, res, feature)
    lin_grad = full_grad["params"][f"layers_{out_index}"]["linear"]["kernel"]
    layer_score = layer_baseline(A, G, lin_grad)
    return rscore, layer_score, loss_sqnorm
