import flax.linen as nn
import jax


def sg(x):
    return jax.lax.stop_gradient(x)


def st(x):
    return x - sg(x)


def maybe_remat(module, enabled):
    if enabled:
        return nn.remat(module)
    else:
        return module
