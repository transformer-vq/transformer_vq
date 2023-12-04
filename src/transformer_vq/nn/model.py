import dataclasses

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.linen import partitioning as nn_partitioning

from transformer_vq.nn.attn import get_attn_cls
from transformer_vq.nn.mlp import MultiLayerPerceptron
from transformer_vq.nn.norm import RMSNorm
from transformer_vq.nn.pe import ScaledSinusoidalEmbs
from transformer_vq.nn.sharding import sharding_constraint
from transformer_vq.nn.types import TransformerConfig


class TransformerLayer(nn.Module):
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

    @nn.compact
    def __call__(self, inputs, _):
        x, l_commit, l_codebook = inputs
        attn_cls = get_attn_cls(self.config.attn_type)
        mlp_cls = attn_cls if self.config.head_type == "shga" else MultiLayerPerceptron
        pspec_tuple = ("data",) + (None,) * (x.ndim - 1)
        n_attn_layer = self.config.n_layer
        if self.config.head_type == "shga":
            n_attn_layer *= 2

        out0 = attn_cls(self.config, self.global_mesh)(x)
        x += out0.pop("res")
        x = sharding_constraint(x, self.global_mesh, pspec_tuple)
        if self.config.attn_type.startswith("vq"):
            l_commit += out0.pop("l_commit")
            l_codebook += out0.pop("l_codebook")

        out1 = mlp_cls(self.config, self.global_mesh)(x)
        x += out1.pop("res")
        x = sharding_constraint(x, self.global_mesh, pspec_tuple)
        if self.config.attn_type.startswith("vq") and self.config.head_type == "shga":
            l_commit += out1.pop("l_commit")
            l_codebook += out1.pop("l_codebook")
        return (x, l_commit, l_codebook), None


class Transformer(nn.Module):
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

    def setup(self):
        self.apply_config()
        self.token_embed = nn.Embed(
            self.n_vocab,
            self.d_model,
            param_dtype=jnp.float32,
            dtype=jnp.float32,
            embedding_init=nn.with_partitioning(
                self.e_init, names=(None, None), mesh=self.global_mesh
            ),
        )
        self.pos_embed = ScaledSinusoidalEmbs(self.config, self.global_mesh)
        self.stack = nn.scan(
            nn_partitioning.remat(TransformerLayer),
            length=self.n_layer,
            variable_axes=dict(params=0),
            variable_broadcast=False,
            split_rngs=dict(params=True),
            metadata_params={nn.PARTITION_NAME: None},
        )(config=self.config, global_mesh=self.global_mesh)
        self.out_ln = RMSNorm()
        if not self.e_tie:
            self.out_proj = nn.Dense(
                self.n_vocab,
                kernel_init=nn.with_partitioning(
                    self.w_init, names=(None, None), mesh=self.global_mesh
                ),
                use_bias=False,
                param_dtype=jnp.float32,
                dtype=jnp.float32,
            )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def __call__(self, inputs):
        chex.assert_shape(inputs, [None, self.sequence_len])
        inputs = sharding_constraint(inputs, self.global_mesh, ("data", None))
        x = self.token_embed(inputs).astype(self.dtype)
        x = sharding_constraint(x, self.global_mesh, ("data", None, None))
        if self.pe_abs:
            x += self.pos_embed(length=self.sequence_len, offset=0)

        x = sharding_constraint(x, self.global_mesh, ("data", None, None))
        x = get_attn_cls(self.attn_type).pre_reshape(x, self.config)
        x = sharding_constraint(
            x, self.global_mesh, ("data",) + (None,) * (x.ndim - 2) + (None,)
        )

        l_commit = jnp.zeros([], dtype=self.dtype)
        l_codebook = jnp.zeros([], dtype=self.dtype)
        (x, l_commit, l_codebook), _ = self.stack((x, l_commit, l_codebook), None)

        x = sharding_constraint(
            x, self.global_mesh, ("data",) + (None,) * (x.ndim - 2) + (None,)
        )
        x = get_attn_cls(self.attn_type).post_reshape(x, self.config)
        x = sharding_constraint(x, self.global_mesh, ("data", None, None))

        if self.e_preln:
            x = self.out_ln(x)
        x = sharding_constraint(x, self.global_mesh, ("data", None, None))
        x *= self.e_scale
        x = sharding_constraint(x, self.global_mesh, ("data", None, None))
        if self.e_tie:
            x = self.token_embed.attend(x)
        else:
            x = self.out_proj(x)
        x = sharding_constraint(x, self.global_mesh, ("data", None, None))
        chex.assert_shape(x, [None, self.sequence_len, self.n_vocab])
        return dict(
            logits=x,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=None,
        )
