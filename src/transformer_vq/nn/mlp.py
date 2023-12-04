import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_vq.nn.norm import RMSNorm
from transformer_vq.nn.sharding import sharding_constraint
from transformer_vq.nn.types import TransformerConfig


class MultiLayerPerceptron(nn.Module):
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

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
        time = [None] * len(x.ndim - 2)
        proj_kwargs = dict(
            param_dtype=self.config.param_dtype,
            dtype=self.config.dtype,
            use_bias=False,
        )
        x = sharding_constraint(x, self.global_mesh, ("data", *time, None))
        x = RMSNorm()(x)
        x = sharding_constraint(x, self.global_mesh, ("data", *time, None))
        x = nn.Dense(
            MultiLayerPerceptron.get_d_ff(self.config),
            **proj_kwargs,
            kernel_init=nn.with_partitioning(
                self.config.w_init, names=(None, "model"), mesh=self.global_mesh
            ),
        )(x)
        x = sharding_constraint(x, self.global_mesh, ("data", *time, "model"))
        x = jnp.square(jax.nn.relu(x))
        x = sharding_constraint(x, self.global_mesh, ("data", *time, "model"))
        x = nn.Dense(
            self.config.d_model,
            **proj_kwargs,
            kernel_init=nn.with_partitioning(
                self.config.w_init, names=("model", None), mesh=self.global_mesh
            ),
        )(x)
        x = sharding_constraint(x, self.global_mesh, ("data", *time, None))
        return dict(res=x)
