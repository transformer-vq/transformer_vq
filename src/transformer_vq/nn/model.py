import dataclasses

import chex
import flax.linen as nn
import jax.numpy as jnp

from transformer_vq.nn.attn2 import get_attn_cls
from transformer_vq.nn.mlp import MultiLayerPerceptron
from transformer_vq.nn.norm import RMSNorm
from transformer_vq.nn.pe import ScaledSinusoidalEmbs
from transformer_vq.nn.types import TransformerConfig


class TransformerLayer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        x, l_commit, l_codebook = inputs
        attn_cls = get_attn_cls(self.config.attn_type)
        mlp_cls = attn_cls if self.config.head_type == "shga" else MultiLayerPerceptron
        layerdrop_kwargs = dict(
            rng_collection="timeless",
            deterministic=not self.config.is_train,
            broadcast_dims=[i for i in range(1, x.ndim)],  # broadcast except over batch
        )

        out0 = attn_cls(self.config)(x)
        x += nn.Dropout(self.config.p_droplyr, **layerdrop_kwargs)(out0.pop("res"))
        if self.config.attn_type.startswith("vq"):
            l_commit += out0.pop("l_commit")
            l_codebook += out0.pop("l_codebook")

        out1 = mlp_cls(self.config)(x)
        x += nn.Dropout(self.config.p_droplyr, **layerdrop_kwargs)(out1.pop("res"))
        if self.config.attn_type.startswith("vq") and self.config.head_type == "shga":
            l_commit += out1.pop("l_commit")
            l_codebook += out1.pop("l_codebook")
        return x, l_commit, l_codebook


class Transformer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        self.token_embed = nn.Embed(
            self.n_vocab,
            self.d_model,
            param_dtype=jnp.float32,
            dtype=jnp.float32,
            embedding_init=self.e_init,
        )
        self.pos_embed = ScaledSinusoidalEmbs(self.config)
        self.stack = nn.remat_scan(TransformerLayer, lengths=(self.n_layer, 1))(
            config=self.config, name="stack"
        )
        self.out_ln = RMSNorm()
        if not self.e_tie:
            self.out_proj = nn.Dense(
                self.n_vocab,
                kernel_init=self.w_init,
                bias_init=self.b_init,
                use_bias=True,
                param_dtype=jnp.float32,
                dtype=jnp.float32,
            )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def __call__(self, inputs):
        chex.assert_shape(inputs, [None, self.sequence_len])
        x = self.token_embed(inputs).astype(self.dtype)
        l_commit = jnp.zeros([], dtype=self.dtype)
        l_codebook = jnp.zeros([], dtype=self.dtype)
        if self.pe_abs:
            x += self.pos_embed(length=self.sequence_len, offset=0)
        x = get_attn_cls(self.attn_type).pre_reshape(x, self.config)
        x, l_commit, l_codebook = self.stack((x, l_commit, l_codebook))
        x = get_attn_cls(self.attn_type).post_reshape(x, self.config)
        if self.e_preln:
            x = self.out_ln(x)
        x *= self.e_scale
        if self.e_tie:
            x = self.token_embed.attend(x)
        else:
            x = self.out_proj(x)
        chex.assert_shape(x, [None, self.sequence_len, self.n_vocab])
        return dict(
            logits=x,
            l_commit=l_commit,
            l_codebook=l_codebook,
        )
