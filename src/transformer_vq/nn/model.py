import dataclasses

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_vq.nn.attn import VQAttention
from transformer_vq.nn.emb import Embeddings
from transformer_vq.nn.norm import LayerNorm
from transformer_vq.nn.pe import ScaledSin
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import VQSpec


class TransformerLayer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        attn_scan_args = dict(
            variable_broadcast="params",
            split_rngs=dict(
                params=False,
                timeless=False,
                ephemeral=True,
            ),
            in_axes=0,
            out_axes=0,  # metrics are zero-dimensional, so have to stack on axis 0
        )
        self.scanned_attn1 = nn.scan(VQAttention, **attn_scan_args)(self.config)
        self.scanned_attn2 = nn.scan(VQAttention, **attn_scan_args)(self.config)

        drop_kwargs = dict(
            rng_collection="timeless",
            deterministic=not self.is_train,
            broadcast_dims=(0, 2, 3),  # broadcast over all axes except batch
        )
        self.droplyr1 = nn.Dropout(self.p_droplyr, **drop_kwargs)
        self.droplyr2 = nn.Dropout(self.p_droplyr, **drop_kwargs)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def initial_state(config, batch_size):
        return [
            VQAttention.initial_state(config=config, batch_size=batch_size),
            VQAttention.initial_state(config=config, batch_size=batch_size),
        ]

    def __call__(self, x, doc_ids, state, vq_spec):
        n_block, batch_size, *_ = x.shape
        dims = chex.Dimensions(
            K=n_block,
            B=batch_size,
            L=self.block_len,
            D=self.d_model,
        )
        state1, state2 = state

        chex.assert_shape(x, dims["KBLD"])
        attn1_input_dict = dict(input_features=x, doc_ids=doc_ids, vq_spec=vq_spec)
        attn1_state, attn1_output_dict = self.scanned_attn1(state1, attn1_input_dict)
        r1 = attn1_output_dict.pop("res")
        chex.assert_shape(r1, dims["KBLD"])
        x += self.droplyr1(r1)
        attn1_output_dict = jax.tree_util.tree_map(jnp.mean, attn1_output_dict)

        attn2_input_dict = dict(input_features=x, doc_ids=doc_ids, vq_spec=vq_spec)
        attn2_state, attn2_output_dict = self.scanned_attn2(state2, attn2_input_dict)
        r2 = attn2_output_dict.pop("res")
        chex.assert_shape(r2, dims["KBLD"])
        x += self.droplyr2(r2)
        attn2_output_dict = jax.tree_util.tree_map(jnp.mean, attn2_output_dict)

        l_commit = attn1_output_dict.pop("l_commit")
        l_commit += attn2_output_dict.pop("l_commit")
        l_codebook = attn1_output_dict.pop("l_codebook")
        l_codebook += attn2_output_dict.pop("l_codebook")
        metric_dict = jax.tree_util.tree_map(
            lambda a, b: (a + b) / 2,
            attn1_output_dict.pop("metrics"),
            attn2_output_dict.pop("metrics"),
        )
        return dict(
            output_features=x,
            attn_state=[attn1_state, attn2_state],
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metric_dict,
        )


class Transformer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        if not self.no_emb or self.e_tie:
            self.token_embedder = Embeddings(self.config)
        if self.pe_abs:
            self.position_embedder = ScaledSin(self.config)
        self.transformer_layers = [
            nn.remat(TransformerLayer)(self.config) for _ in range(self.n_layer)
        ]
        if self.e_preln:
            self.out_ln = LayerNorm(self.d_model, self.param_dtype)
        if not self.e_tie:
            self.out_proj = nn.Dense(
                self.n_vocab,
                use_bias=True,
                kernel_init=self.w_init,
                bias_init=self.b_init,
                param_dtype=self.param_dtype,
                dtype=self.param_dtype,  # always use full-precision logits
            )
        drop_kwargs = dict(rng_collection="ephemeral", deterministic=not self.is_train)
        self.dropemb = nn.Dropout(self.p_dropemb, **drop_kwargs)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def initial_state(config, batch_size):
        return [
            TransformerLayer.initial_state(
                config=config,
                batch_size=batch_size,
            )
            for _ in range(config.n_layer)
        ]

    def get_chex_dims(self, batch_size, present_len):
        return chex.Dimensions(
            B=batch_size,
            P=present_len,
            K=present_len // self.block_len,
            L=self.block_len,
            D=self.d_model,
            V=self.n_vocab,
        )

    def get_blocks_from_sequence(self, x):
        batch_size, present_len, *suffix = x.shape
        n_block = present_len // self.block_len
        x = jnp.reshape(x, [batch_size, n_block, self.block_len, *suffix])
        suffix_axes = list(range(3, x.ndim))
        x = jnp.transpose(x, (1, 0, 2, *suffix_axes))
        return x

    def get_sequence_from_blocks(self, x):
        num_block, batch_size, block_len, *suffix = x.shape
        suffix_axes = list(range(3, x.ndim))
        x = jnp.transpose(x, (1, 0, 2, *suffix_axes))
        x = jnp.reshape(x, [batch_size, num_block * block_len, *suffix])
        return x

    def get_blocks_of_vq_spec(self, vq_spec):
        if vq_spec is None:
            return None

        chex.assert_rank(vq_spec.loss_mask, 2)
        n_block = vq_spec.loss_mask.shape[1] // self.block_len

        def expand_and_tile(array):
            mult = [n_block] + [1 for _ in range(jnp.ndim(array))]
            return jnp.tile(array[None, ...], mult)

        return VQSpec.create(
            n_device=expand_and_tile(vq_spec.n_device),
            n_block_per_update=expand_and_tile(vq_spec.n_block_per_update),
            loss_mask=jnp.transpose(
                jnp.reshape(vq_spec.loss_mask, [-1, n_block, self.block_len]),
                (1, 0, 2),
            ),
        )

    @staticmethod
    def maybe_aggregate(accumulator_dict, new_dict):
        # old is empty
        if len(accumulator_dict) == 0:
            return new_dict
        # new is empty
        if len(new_dict) == 0:
            return accumulator_dict
        # old is nonempty, new is nonempty
        return jax.tree_util.tree_map(lambda a, b: a + b, accumulator_dict, new_dict)

    @staticmethod
    def average_layer_metrics(aux, n_layer):
        if "metrics" not in aux:
            return aux
        metrics = aux.pop("metrics")
        metrics = jax.tree_util.tree_map(lambda y: y / n_layer, metrics)
        new_aux = dict(metrics=metrics, **aux)
        return new_aux

    def __call__(self, inputs, doc_ids, state, vq_spec):
        batch_size, present_len, *_ = inputs.shape
        dims = self.get_chex_dims(batch_size, present_len)
        chex.assert_shape(doc_ids, dims["BP"])
        new_state = []
        aux = {}
        x = inputs
        if not self.no_emb:
            x = self.token_embedder(x)
        if self.pe_abs:
            offset = state[0][0]["pos_offset"]
            x += self.position_embedder(length=present_len, offset=offset)
        x = self.dropemb(x)
        x = self.get_blocks_from_sequence(x)
        doc_ids = self.get_blocks_from_sequence(doc_ids)
        vq_spec = self.get_blocks_of_vq_spec(vq_spec)
        chex.assert_shape(x, dims["KBLD"])
        for i in range(self.n_layer):
            layer_output_dict = self.transformer_layers[i](
                x=x, doc_ids=doc_ids, state=state[i], vq_spec=vq_spec
            )
            new_state.append(layer_output_dict.pop("attn_state"))
            x = layer_output_dict.pop("output_features")
            chex.assert_shape(x, dims["KBLD"])
            aux = Transformer.maybe_aggregate(aux, layer_output_dict)
        x = self.get_sequence_from_blocks(x)
        aux = Transformer.average_layer_metrics(aux, self.n_layer)
        if self.e_preln:
            x = self.out_ln(x)
        x = self.token_embedder.logits(x) if self.e_tie else self.out_proj(x)
        x *= self.e_scale
        x = jax.nn.log_softmax(x, axis=-1)
        chex.assert_shape(x, dims["BPV"])
        return dict(logprobs=x, attn_state=new_state, **aux)
