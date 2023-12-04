import dataclasses

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_vq.nn.sharding import sharding_constraint
from transformer_vq.nn.types import TransformerConfig


def get_sinusoid_embs(length, width, lam, flip, dtype, global_mesh, start=0):
    pos_seq = start + jnp.arange(length)
    chex.assert_shape(pos_seq, [length])
    inv_lams = 1 / (lam ** (jnp.arange(0, width, 2) / width))
    pre = pos_seq[..., None] * inv_lams[None, ...]
    pre = sharding_constraint(pre, global_mesh, (None, None))
    sin = jnp.sin(pre).astype(dtype)
    cos = jnp.cos(pre).astype(dtype)
    cat = jnp.concatenate([sin, cos], axis=-1)
    chex.assert_shape(cat, [length, width])
    if not flip:
        return cat
    return jnp.flip(cat, axis=0)


class ScaledSinusoidalEmbs(nn.Module):
    # see w. hua et al., 2022
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

    def setup(self):
        self.apply_config()
        self.scale = self.param(
            "scale",
            nn.with_partitioning(
                jax.nn.initializers.zeros, names=(None,), mesh=self.global_mesh
            ),
            [],
            jnp.float32,
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def __call__(self, length, offset):
        embs = get_sinusoid_embs(
            length=length,
            start=offset,
            width=self.d_model,
            lam=self.pe_lam,
            flip=False,
            dtype=self.dtype,
            global_mesh=self.global_mesh,
        )
        return self.scale.astype(self.dtype) * embs


class XLBiasProducer(nn.Module):
    # see z. dai et al., 2019
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

    def setup(self):
        self.apply_config()
        assert self.d_model % self.d_k == 0
        self.tau = self.d_k**0.5
        proj_kwargs = dict(
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        assert self.head_type == "shga"
        self.xl_r_proj = nn.Dense(
            self.d_k,
            **proj_kwargs,
            kernel_init=nn.with_partitioning(
                self.w_init, names=(None, None), mesh=self.global_mesh
            ),
        )
        self.xl_u = self.param(
            "xl_u",
            nn.with_partitioning(
                jax.nn.initializers.zeros, names=(None,), mesh=self.global_mesh
            ),
            [self.d_k],
            self.param_dtype,
        )
        self.xl_v = self.param(
            "xl_v",
            nn.with_partitioning(
                jax.nn.initializers.zeros, names=(None,), mesh=self.global_mesh
            ),
            [self.d_k],
            self.param_dtype,
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def rel_shift(x, head_type, global_mesh):
        assert head_type == "shga"
        *prefix, q_len, k_len = x.shape
        nones = [None] * (len(prefix) + 1)
        pad_spec = [(0, 0)] * len(prefix) + [(0, 0), (1, 0)]
        x = jnp.pad(x, pad_spec)
        x = sharding_constraint(x, global_mesh, ("data", *nones))
        x = jnp.reshape(x, [*prefix, k_len + 1, q_len])
        x = sharding_constraint(x, global_mesh, ("data", *nones))
        x = x[..., 1:, :]
        x = sharding_constraint(x, global_mesh, ("data", *nones))
        x = jnp.reshape(x, [*prefix, q_len, k_len])
        x = sharding_constraint(x, global_mesh, ("data", *nones))
        return x

    @staticmethod
    def get_causal_keep_mask(
        q_len,
        k_len,
        with_locality,
        global_mesh,
        with_alloc=False,
        invalid_len=None,
    ):
        i = jnp.arange(q_len)[..., None]
        j = jnp.arange(k_len)[None, ...]
        causal_mask = jnp.less_equal(j - (k_len - q_len), i)
        keep_mask = causal_mask
        if with_alloc:
            assert invalid_len is not None
            alloc_mask = jnp.greater_equal(j, jnp.array([invalid_len])[None, ...])
            keep_mask = jnp.logical_and(alloc_mask, causal_mask)
        if with_locality:
            window_mask = jnp.greater_equal(j, i)
            keep_mask = jnp.logical_and(keep_mask, window_mask)
        keep_mask = sharding_constraint(keep_mask, global_mesh, (None, None))
        return keep_mask

    def __call__(self, k_len):
        xl_r = get_sinusoid_embs(
            length=k_len,
            width=self.d_model,
            lam=self.pe_lam,
            flip=True,
            dtype=self.dtype,
            global_mesh=self.global_mesh,
        )
        xl_r = self.xl_r_proj(xl_r)
        xl_u = self.xl_u.astype(self.dtype)
        xl_v = self.xl_v.astype(self.dtype)
        mult = jnp.array([self.tau**-0.5], dtype=self.dtype)
        return map(lambda y: mult * y, [xl_r, xl_u, xl_v])
