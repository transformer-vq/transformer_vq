import dataclasses

import chex
import flax.linen as nn
import jax.numpy as jnp

from transformer_vq.nn.types import TransformerConfig


def get_sinusoid_embs(length, width, lam, flip, start=0):
    pos_seq = start + jnp.arange(length)
    chex.assert_shape(pos_seq, [length])
    inv_lams = 1 / (lam ** (jnp.arange(0, width, 2) / width))
    pre = pos_seq[..., None] * inv_lams[None, ...]
    sin = jnp.sin(pre)
    cos = jnp.cos(pre)
    cat = jnp.concatenate([sin, cos], axis=-1)
    chex.assert_shape(cat, [length, width])
    if not flip:
        return cat
    return jnp.flip(cat, axis=0)


class ScaledSin(nn.Module):
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
            length=length, start=offset, width=self.d_model, lam=self.pe_lam, flip=False
        )
        return (self.scale * embs).astype(self.dtype)
