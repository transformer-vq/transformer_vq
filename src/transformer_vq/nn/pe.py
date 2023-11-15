import dataclasses

import chex
import flax.linen as nn
import jax.numpy as jnp

from transformer_vq.nn.types import TransformerConfig


def get_sinusoid_embs(length, width, lam, flip, dtype, start=0):
    pos_seq = start + jnp.arange(length)
    chex.assert_shape(pos_seq, [length])
    inv_lams = 1 / (lam ** (jnp.arange(0, width, 2) / width))
    pre = pos_seq[..., None] * inv_lams[None, ...]
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

    def setup(self):
        self.apply_config()
        self.scale = self.param("scale", self.b_init, [], jnp.float32)

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
        )
        return self.scale.astype(self.dtype) * embs


class XLBiasProducer(nn.Module):
    # see z. dai et al., 2019
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        assert self.d_model % self.d_k == 0
        self.tau = self.d_k**0.5
        proj_kwargs = dict(
            kernel_init=self.w_init,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        q_ch = self.d_model if self.head_type in {"mha", "mqa"} else self.d_k
        k_ch = self.d_model if self.head_type == "mha" else self.d_k
        self.xl_r_proj = nn.Dense(k_ch, **proj_kwargs)
        self.xl_u = self.param("xl_u", self.b_init, [q_ch], self.param_dtype)
        self.xl_v = self.param("xl_v", self.b_init, [q_ch], self.param_dtype)
        self.dropsin = nn.Dropout(
            self.p_dropsin, rng_collection="timeless", deterministic=not self.is_train
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def rel_shift(x):
        *prefix, q_len, k_len = x.shape
        pad_spec = [(0, 0)] * len(prefix) + [(0, 0), (1, 0)]
        x = jnp.pad(x, pad_spec)
        x = jnp.reshape(x, [*prefix, k_len + 1, q_len])
        x = x[..., 1:, :]
        x = jnp.reshape(x, [*prefix, q_len, k_len])
        return x

    @staticmethod
    def get_causal_keep_mask(
        q_len,
        k_len,
        with_locality,
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
        return keep_mask

    def __call__(self, k_len):
        xl_r = get_sinusoid_embs(
            length=k_len,
            width=self.d_model,
            lam=self.pe_lam,
            flip=True,
            dtype=self.dtype,
        )
        xl_r = self.dropsin(xl_r)
        xl_r = self.xl_r_proj(xl_r)
        xl_r = jnp.reshape(xl_r, [k_len, -1, self.d_k])  # WhK
        xl_r = jnp.transpose(xl_r, (1, 0, 2))  # hWK
        xl_r = xl_r * (self.tau**-0.5)
        xl_u = jnp.reshape(self.xl_u, [1, -1, 1, self.d_k]) * (self.tau**-0.5)
        xl_v = jnp.reshape(self.xl_v, [1, -1, 1, self.d_k]) * (self.tau**-0.5)
        return xl_r, xl_u.astype(self.dtype), xl_v.astype(self.dtype)
