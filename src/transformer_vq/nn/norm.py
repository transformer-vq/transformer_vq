import flax.linen as nn
import jax
import jax.numpy as jnp


class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, eps=1e-6):
        x *= jax.lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=-1, keepdims=True))
        return x
