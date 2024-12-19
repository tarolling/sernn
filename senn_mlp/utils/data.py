from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from itertools import islice
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax.tree_util import Partial
from utils.nets import DDense, Layer, Layers, Rational1D
from tqdm import tqdm
from utils.jaxutils import key_iter


def trfm(size, img, channels=1):
    tensor = tf.convert_to_tensor(img)
    # add batch dim
    tensor: tf.Tensor = tf.expand_dims(tensor, axis=0)

    # resize using bilinear interpolation (default `resize` behavior)
    resized_tensor: tf.Tensor = tf.image.resize(tensor, size=[size, size])

    # remove batch dim
    resized_tensor: tf.Tensor = tf.squeeze(resized_tensor, axis=0)

    # convert to numpy array and transpose
    numpy_img = tf.make_ndarray(resized_tensor).transpose(1, 2, 0)

    out = jnp.array(numpy_img)
    if out.shape[-1] != channels:
        if channels == 1:
            out = out.mean(axis=-1, keepdims=True)
        else:
            assert (
                out.shape[-1] == 1
            ), f"incompatible channel number: expected {channels} but got {out.shape[-1]}"
            out = jnp.tile(out, (3,))
    return out


def get_dataset(root, name: str, train, resolution):
    if name == "mnist":
        return MNIST(
            root, train=train, download=True, transform=partial(trfm, resolution)
        )
    elif name == "fmnist":
        return FashionMNIST(
            root, train=train, download=True, transform=partial(trfm, resolution)
        )
    elif name == "cifar10":
        return CIFAR10(
            root, train=train, download=True, transform=partial(trfm, resolution)
        )
    else:
        raise NotImplementedError(f"Dataset '{name}' not recognised.")


def get_chunk(dataset, labels, remap, N, start=0):
    elems = tqdm(
        ((d, remap(l)) for d, l in iter(dataset) if l in labels),
        total=N,
        desc="Compiling data tranch",
    )
    imgs, labels = map(lambda gen: jnp.array(list(gen)), zip(*islice(elems, start, N)))
    return imgs, labels


def cfg_tranch(defaults, tranch, resolution):
    def get(key):
        return tranch[key] if key in tranch else defaults[key].get()

    N = get("N")
    TN = get("TN")
    classes = get("classes")

    dataset = get("dataset")
    root = get("root")
    remap_val = get("remap")
    remap = lambda x: x if remap_val is None else remap_val[classes.index(x)]

    train = get_chunk(
        get_dataset(root, dataset, True, resolution), classes, remap, len(classes) * N
    )
    test = get_chunk(
        get_dataset(root, dataset, False, resolution), classes, remap, len(classes) * TN
    )

    return train, test


def cfg_tranches(cfg, resolution):
    defaults = cfg["defaults"]
    return list(
        cfg_tranch(defaults, tranch, resolution) for tranch in cfg["tranches"].get()
    )


def tranch_cat(tranches, index, train):
    tups = list(
        islice([tranch[0] if train else tranch[1] for tranch in tranches], index + 1)
    )
    return tuple([jnp.concatenate(arrs, axis=0) for arrs in zip(*tups)])


@dataclass
class ModelTemplate:
    capacities: Sequence[int]
    contents: Sequence[Optional[int]]
    rational: bool = False

    def layer(self, features, final=False):
        if self.rational:
            nonlin = partial(Rational1D, residual=True, init_identity=False)
        else:
            nonlin = lambda: jax.nn.silu
        return Layer(features, [], DDense, [] if final else [nonlin])

    def build(self):
        hidden_layers = [self.layer(f) for f in self.capacities[:-1]]
        final_layer = self.layer(self.capacities[-1], final=True)
        return Layers(hidden_layers + [final_layer])

    def enabled_layers(self):
        return list([i for i, d in enumerate(self.contents) if d is not None])

    def disabled_layers(self):
        return list([i for i, d in enumerate(self.contents) if d is None])

    def in_out_indices(self, layer_index):
        conarr = np.array(self.contents)
        split_at = layer_index + 1
        (preceding_enabled,) = np.nonzero(conarr[:split_at][:-1] != None)
        in_index = 0 if len(preceding_enabled) == 0 else preceding_enabled[-1] + 1
        # in_index = layer_index
        (subsequent_enabled,) = np.nonzero(conarr[split_at:] != None)
        out_index = split_at + subsequent_enabled[0]
        return in_index, out_index


class Task(ABC):

    @abstractmethod
    def get_data(cfg, test=False):
        pass
        return dataset, labels

    @abstractmethod
    def loss_function(label, output):
        pass


class Regression(Task):

    @staticmethod
    def loss_function(label, output):
        return ((output - label) ** 2).mean()


class Classification(Task):

    @staticmethod
    def loss_function(label, output):
        return jax.nn.logsumexp(output) - output[label]


class SameFamilyRegression(Regression):

    def __init__(self, cfg, key):
        self.out_size = cfg["task"]["out_size"].get()
        self.in_size = cfg["task"]["in_size"].get(self.out_size)
        capacities = cfg["task"]["hidden"].get() + [self.out_size]
        self.template = ModelTemplate(capacities, capacities, rational=True)
        self.model = self.template.build()
        self.state = self.model.init(key, jnp.zeros((self.in_size,)))
        self.target_func = jax.jit(Partial(self.model.apply, self.state))

        self.N = cfg["data"]["N"].get()
        self.TN = cfg["data"]["TN"].get(self.N)

    def get_data(self, cfg, test=False, key=None):
        assert key is not None
        N = self.TN if test else self.N
        xs = jax.random.normal(key, (N, self.in_size))
        ys = jax.vmap(self.target_func)(xs)
        return xs, ys


class ImgVecClass(Classification):

    def __init__(self, cfg):
        self.cfg = cfg
        self.tranches = cfg_tranches(cfg["data"], cfg["task"]["resolution"].get())

    def get_data(self, _, test=False, index=0):
        dataset, labels = tranch_cat(self.tranches, index, train=not test)
        return jax.vmap(jnp.ravel)(dataset), labels


def process_task(cfg) -> tuple:
    task_type = cfg["task"]["type"].get("regression")
    if task_type == "regression":
        seed = cfg["task"]["seed"].get(cfg["meta"]["seed"].get())
        key = key_iter(seed)
        task = SameFamilyRegression(cfg, next(key))
        train = task.get_data(cfg, test=False, key=next(key))
        test = task.get_data(cfg, test=True, key=next(key))
        out_size = task.out_size
        return task, train, test, out_size
    elif task_type == "classification":
        task = ImgVecClass(cfg)
        train = task.get_data(cfg, test=False)
        test = task.get_data(cfg, test=True)
        out_size = test[1].max() + 1
        return task, train, test, out_size

    else:
        raise ValueError(f"unrecognised task type: {task_type}")
