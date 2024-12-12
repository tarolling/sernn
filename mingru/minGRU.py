# https://arxiv.org/abs/2410.01201v1

from typing import Any

import flax.linen as nn
import jax
import tensorflow as tf
from jax import numpy as jnp
import keras


def exists(v: Any) -> bool:
    return v is not None


def logcumsumexp(x: tf.Tensor, axis: int) -> tf.Tensor:
    max_x = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.math.log(tf.cumsum(tf.exp(x - max_x), axis=axis)) + max_x


# appendix B
# https://github.com/glassroom/heinsen_sequence


def heinsen_associative_scan_log(
    log_coeffs: tf.Tensor, log_values: tf.Tensor
) -> tf.Tensor:
    a_star: tf.Tensor = tf.cumsum(log_coeffs, axis=1)
    log_h0_plus_b_star = logcumsumexp(log_values - a_star, axis=1)
    log_h = a_star + log_h0_plus_b_star
    return tf.exp(log_h)


class ActivationNetwork(keras.layers.Layer):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = keras.layers.Dense(input_dim * 2, activation="relu")
        self.layer2 = keras.layers.Dense(input_dim, activation="relu")
        self.layer3 = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        original_shape = inputs.shape
        if len(original_shape) > 2:
            inputs = tf.reshape(inputs, (-1, original_shape[-1]))
        result = self.layer3(self.layer2(self.layer1(inputs)))
        result = tf.reshape(result, (original_shape[:-1] + (1,),))
        result = tf.squeeze(result, [-1])
        return result


# appendix B.3


def g(x: tf.Tensor) -> tf.Tensor:
    return tf.where(x >= 0, x + 0.5, tf.nn.sigmoid(x))


def log_g(x: tf.Tensor) -> tf.Tensor:
    return tf.where(x >= 0, tf.math.log((tf.nn.relu(x) + 0.5)), -tf.nn.softplus(-x))


def lerp(a: tf.Tensor, b: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    return a + t * (b - a)


class Identity(keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs


# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive


class minGRU(keras.layers.Layer):
    def __init__(self, input_dim: int, expansion_factor: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.expansion_factor = expansion_factor
        self.to_hidden_and_gate = None
        self.to_out = None
        self.activation_net = None

    def build(self, input_shape):
        dim_inner = int(self.input_dim * self.expansion_factor)
        self.to_hidden_and_gate = keras.layers.Dense(dim_inner * 2, use_bias=False)
        self.to_out = (
            keras.layers.Dense(self.input_dim, use_bias=False)
            if self.expansion_factor != 1.0
            else Identity()
        )
        self.activation_net = ActivationNetwork(dim_inner)
        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        prev_hidden: tf.Tensor = None,
        return_next_prev_hidden: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        seq_len = inputs.shape[1]
        hidden, gate = tf.split(self.to_hidden_and_gate(inputs), 2, axis=-1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = self.activation_net(gate)
            out = (
                lerp(prev_hidden, hidden, gate)
                if exists(prev_hidden)
                else hidden * gate
            )
        else:
            # parallel
            log_coeffs: tf.Tensor = -tf.nn.softplus(gate)
            log_z = -tf.nn.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = tf.concat([prev_hidden, log_values], axis=1)
                log_coeffs = tf.pad(log_coeffs, [[0, 0], [1, 0], [0, 0]])

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]
        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden
