import jax
import jax.numpy as jnp


def glu(x):
    xf, xg = jnp.split(x, 2, axis=-1)
    return xf * jax.nn.sigmoid(xg)


def swiglu(x):
    xf, xg = jnp.split(x, 2, axis=-1)
    return xf * jax.nn.silu(xg)


def sqrelu(x):
    return jnp.square(jax.nn.relu(x))


def get_activation(name):
    if name == "relu":
        return jax.nn.relu
    if name == "gelu":
        return jax.nn.gelu
    if name == "silu":
        return jax.nn.silu
    if name == "swiglu":
        return swiglu
    if name == "sqrelu":
        return sqrelu
    raise NotImplementedError
