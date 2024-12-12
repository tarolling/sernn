from typing import Any

import flax.linen as nn
import jax
import tensorflow as tf
from jax import numpy as jnp
import keras

from minGRU import minGRU


def exists(v: Any) -> bool:
    return v is not None


def default(v: Any, d: Any) -> Any:
    return v if exists(v) else d


# classes
class RMSNorm(keras.layers.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.scale = input_dim**0.5
        self.gamma = self.add_weight((input_dim,), initializer="zeros")

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1) * self.scale * (self.gamma + 1)


# conv
class CausalDepthWiseConv1d(keras.layers.Layer):
    def __init__(self, input_dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = keras.Sequential()
        self.net.add(
            keras.layers.Conv1D(
                filters=input_dim, kernel_size=kernel_size, groups=input_dim
            )
        )
        self.net.add(keras.layers.Conv1D(filters=input_dim, kernel_size=1))

    def call(self, inputs):
        inputs = tf.transpose(inputs, perm=[1, 2])  # b n d -> b d n
        inputs = jnp.pad(inputs, pad_width=(self.kernel_size - 1, 0))
        inputs = self.net(inputs)
        return tf.transpose(inputs, perm=[1, 2])  # b d n -> b n d


# main class


class minGRULM(keras.models.Sequential):
    def __init__(
        self,
        *,
        num_tokens,
        input_dim,
        depth,
        ff_mult=4,
        min_gru_expansion=1.5,
        conv_kernel_size=3,
        enable_conv=False
    ):
        super().__init__()
        self.add(keras.layers.Input(shape=(input_dim,)))
        self.add(keras.layers.Embedding(input_dim=num_tokens, output_dim=input_dim))

        for _ in range(depth):
            if enable_conv:
                self.add(CausalDepthWiseConv1d(input_dim, conv_kernel_size))
            self.add(RMSNorm(input_dim))
            self.add(minGRU(input_dim, expansion_factor=min_gru_expansion))
            self.add(RMSNorm(input_dim)),
            self.add(keras.layers.Dense(int(input_dim * ff_mult), activation="gelu"))
            self.add(keras.layers.Dense(input_dim))

        self.add(RMSNorm(input_dim))
        self.add(keras.layers.Dense(num_tokens, use_bias=False))
        # self.norm = RMSNorm(input_dim)
        # self.to_logits = keras.layers.Dense((input_dim, num_tokens), use_bias=False)
        self.can_cache = not enable_conv

    def call(
        self, inputs, return_loss=False, return_prev_hiddens=False, prev_hiddens=None
    ):
        if return_loss:
            inputs, labels = inputs[:, :-1], inputs[:, 1:]

        inputs = self.token_emb(inputs)

        if exists(prev_hiddens):
            inputs = inputs[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        return inputs
