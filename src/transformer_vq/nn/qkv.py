import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_vq.nn.norm import RMSNorm
from transformer_vq.nn.types import TransformerConfig


class QKVGProducer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        self.tau = self.d_k**0.5
        self.q_ln = RMSNorm()
        self.k_ln = RMSNorm()
        proj_kwargs = dict(
            kernel_init=self.w_init,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        n_q = QKVGProducer.get_n_q(self.config)
        n_kv = QKVGProducer.get_n_kv(self.config)
        d_k = self.d_k
        d_v = self.get_d_v(self.config)
        self.q_proj = nn.Dense(n_q * d_k, **proj_kwargs)
        self.k_proj = nn.Dense(n_kv * d_k, **proj_kwargs)
        self.v_proj = nn.Dense(n_kv * d_v, **proj_kwargs)
        if self.head_type == "shga":
            self.g_proj = nn.Dense(d_v, **proj_kwargs)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def get_n_q(config):
        if config.head_type == "mha":
            assert config.d_model % config.d_k == 0
            return config.d_model // config.d_k
        if config.head_type == "mqa":
            assert config.d_model % config.d_k == 0
            return config.d_model // config.d_k
        if config.head_type == "shga":
            return 1
        raise NotImplementedError(f"Unrecognized head type {config.head_type}")

    @staticmethod
    def get_n_kv(config):
        if config.head_type == "mha":
            assert config.d_model % config.d_k == 0
            return config.d_model // config.d_k
        if config.head_type == "mqa":
            assert config.d_model % config.d_k == 0
            return 1
        if config.head_type == "shga":
            return 1
        raise NotImplementedError(f"Unrecognized head type {config.head_type}")

    @staticmethod
    def get_d_v(config):
        if config.head_type == "mha":
            return config.d_k
        if config.head_type == "mqa":
            return config.d_k
        if config.head_type == "shga":
            return config.d_model * 2
        raise NotImplementedError(f"Unrecognized head type {config.head_type}")

    def get_queries(self, x_tilde):
        *prefix, _ = x_tilde.shape
        q = self.q_proj(x_tilde)
        q = jnp.reshape(q, [*prefix, -1, self.d_k])
        q = self.q_ln(q) * (self.tau**-0.5)
        return q

    def get_keys_and_values(self, x_tilde):
        *prefix, _ = x_tilde.shape
        k = self.k_proj(x_tilde)
        v = self.v_proj(x_tilde)
        k = jnp.reshape(k, [*prefix, -1, self.d_k])
        v = jnp.reshape(v, [*prefix, -1, QKVGProducer.get_d_v(self.config)])
        k = self.k_ln(k) * (self.tau**-0.5)
        if self.head_type == "shga":
            v = jax.nn.silu(v)
        return k, v

    def get_gates(self, x_tilde):
        assert self.head_type == "shga"
        g = self.g_proj(x_tilde)
        g = jax.nn.silu(g)
        return g

    def __call__(self, x_tilde):
        raise NotImplementedError
