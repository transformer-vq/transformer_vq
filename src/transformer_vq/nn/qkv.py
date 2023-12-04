import dataclasses

import flax.linen as nn
import jax

from transformer_vq.nn.norm import RMSNorm
from transformer_vq.nn.sharding import sharding_constraint
from transformer_vq.nn.types import TransformerConfig


class QKVGProducer(nn.Module):
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

    def setup(self):
        self.apply_config()
        assert self.head_type == "shga"
        self.tau = self.d_k**0.5
        self.q_ln = RMSNorm()
        self.k_ln = RMSNorm()
        proj_kwargs = dict(
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.q_proj = nn.Dense(
            self.d_k,
            **proj_kwargs,
            kernel_init=nn.with_partitioning(
                self.w_init, names=(None, None), mesh=self.global_mesh
            ),
        )
        self.k_proj = nn.Dense(
            self.d_k,
            **proj_kwargs,
            kernel_init=nn.with_partitioning(
                self.w_init, names=(None, None), mesh=self.global_mesh
            ),
        )
        self.v_proj = nn.Dense(
            self.get_d_v(self.config),
            **proj_kwargs,
            kernel_init=nn.with_partitioning(
                self.w_init, names=(None, "model"), mesh=self.global_mesh
            ),
        )
        self.g_proj = nn.Dense(
            self.get_d_v(self.config),
            **proj_kwargs,
            kernel_init=nn.with_partitioning(
                self.w_init, names=(None, "model"), mesh=self.global_mesh
            ),
        )
        assert self.head_type == "shga"

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def get_d_v(config):
        if config.head_type == "shga":
            return config.d_model * 2
        raise NotImplementedError(f"Unrecognized head type {config.head_type}")

    def get_queries(self, x_tilde):
        assert self.head_type == "shga"
        time = [None] * (x_tilde.ndim - 2)
        x_tilde = sharding_constraint(x_tilde, self.global_mesh, ("data", *time, None))
        q = self.q_proj(x_tilde)
        q = sharding_constraint(q, self.global_mesh, ("data", *time, None))
        q = self.q_ln(q) * (self.tau**-0.5)
        q = sharding_constraint(q, self.global_mesh, ("data", *time, None))
        return q

    def get_keys_and_values(self, x_tilde):
        assert self.head_type == "shga"
        time = [None] * (x_tilde.ndim - 2)
        x_tilde = sharding_constraint(x_tilde, self.global_mesh, ("data", *time, None))
        k = self.k_proj(x_tilde)
        k = sharding_constraint(k, self.global_mesh, ("data", *time, None))
        v = self.v_proj(x_tilde)
        v = sharding_constraint(v, self.global_mesh, ("data", *time, "model"))
        k = self.k_ln(k) * (self.tau**-0.5)
        k = sharding_constraint(k, self.global_mesh, ("data", *time, None))
        if self.head_type == "shga":
            v = jax.nn.silu(v)
            v = sharding_constraint(v, self.global_mesh, ("data", *time, "model"))
        return k, v

    def get_gates(self, x_tilde):
        assert self.head_type == "shga"
        time = [None] * (x_tilde.ndim - 2)
        x_tilde = sharding_constraint(x_tilde, self.global_mesh, ("data", *time, None))
        g = self.g_proj(x_tilde)
        g = sharding_constraint(g, self.global_mesh, ("data", *time, "model"))
        g = jax.nn.silu(g)
        g = sharding_constraint(g, self.global_mesh, ("data", *time, "model"))
        return g

    def __call__(self, x_tilde):
        raise NotImplementedError
