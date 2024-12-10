import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.example_libraries import optimizers


EPOCHS = 1000


def init_network_params(layer_sizes: list[int], key: jax.Array) -> list:
    keys = random.split(key, len(layer_sizes))
    return [
        {"w": random.normal(k, (m, n)) * jnp.sqrt(2.0 / m), "b": jnp.zeros(n)}
        for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])
    ]


def forward(params, x) -> jax.Array:
    # TODO: make activation more flexible
    preactivations = x
    activations = x
    for layer in params[:-1]:
        activations = jnp.tanh(jnp.dot(activations, layer["w"]) + layer["b"])
    final_layer = params[-1]
    return jnp.dot(activations, final_layer["w"]) + final_layer["b"]


def loss(params, x, y) -> jax.Array:
    logits = forward(params, x)
    return -jnp.mean(y * logits - jnp.logaddexp(0, logits))


@jit
def update(params, x, y, opt_state) -> tuple:
    value, grads = jax.value_and_grad(loss)(params, x, y)
    for g in grads:
        print(f"b: {g['b']}")
        print(f"w: {g['w']}")
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


if __name__ == "__main__":
    # generate random data
    key = random.PRNGKey(0)
    x_key, y_key = random.split(key)
    x = random.normal(x_key, (1000, 5))
    y = random.bernoulli(y_key, 0.5, (1000, 1)).astype(jnp.float32)

    # init network
    layer_sizes = [5, 32, 32, 1]
    params = init_network_params(layer_sizes, random.PRNGKey(1))

    # set up optimizer
    lr = 0.01
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    # training
    for i in range(EPOCHS):
        params, opt_state, loss_value = update(params, x, y, opt_state)
        if i % (EPOCHS / 10) == 0:
            print(f"Step {i}, Loss: {loss_value}")

    # preds
    final_params = get_params(opt_state)
    preds = jax.nn.sigmoid(forward(final_params, x))
    print(f"Final predictions shape: {preds.shape}")
    print(f"Sample preds: {preds[:5]}")
