import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_vq.nn.types import Dtype


class LayerNorm(nn.Module):
    input_dim: int
    param_dtype: Dtype
    center: bool = False  # rms layer norm by default
    norm: bool = True
    gain: bool = True
    bias: bool = True

    def setup(self):
        initializer_args = [[self.input_dim], self.param_dtype]
        if self.gain:
            self.g = self.param("g", jax.nn.initializers.ones, *initializer_args)
        if self.bias:
            self.b = self.param("b", jax.nn.initializers.zeros, *initializer_args)

    def __call__(self, x, eps=1e-6):
        chex.assert_shape(x, (..., self.input_dim))
        dtype = x.dtype
        x = x.astype(jnp.float32)
        if self.center:
            x -= jnp.mean(x, axis=-1, keepdims=True)
        if self.norm:
            x *= jax.lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=-1, keepdims=True))
        broadcast_shape = [1 for _ in range(x.ndim - 1)] + [self.input_dim]
        if self.gain:
            x *= jnp.reshape(self.g, broadcast_shape)
        if self.bias:
            x += jnp.reshape(self.b, broadcast_shape)
        return x.astype(dtype)
