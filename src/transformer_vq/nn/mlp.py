import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_vq.nn.norm import RMSNorm
from transformer_vq.nn.types import TransformerConfig


class MultiLayerPerceptron(nn.Module):
    config: TransformerConfig

    @staticmethod
    def get_d_ff(config):
        if config.head_type == "mha":
            return config.d_model * 4
        if config.head_type == "mqa":
            return config.d_model * 5
        if config.head_type == "shga":
            return 0
        raise NotImplementedError(f"Unrecognized head type {config.head_type}")

    @nn.compact
    def __call__(self, x):
        proj_kwargs = dict(
            kernel_init=self.config.w_init,
            param_dtype=self.config.param_dtype,
            dtype=self.config.dtype,
            use_bias=False,
        )
        x = RMSNorm()(x)
        x = nn.Dense(MultiLayerPerceptron.get_d_ff(self.config), **proj_kwargs)(x)
        x = jnp.square(jax.nn.relu(x))
        x = nn.Dense(self.config.d_model, **proj_kwargs)(x)
        return dict(res=x)
