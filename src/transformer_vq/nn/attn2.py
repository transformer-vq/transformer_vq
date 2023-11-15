import dataclasses

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_vq.nn.attn1 import get_attn_cls_old
from transformer_vq.nn.attn1 import ScannedAttnOld
from transformer_vq.nn.norm import RMSNorm
from transformer_vq.nn.pe import XLBiasProducer
from transformer_vq.nn.qkv import QKVGProducer
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import VectorQuantizer

INFTY_APPROX = 1e30  # large value approximating infinity


class VQAttention(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        assert self.sequence_len // self.block_len > 2
        self.tau = self.d_k**0.5
        self.input_ln = RMSNorm()
        self.qkvg_producer = QKVGProducer(self.config)
        self.quantizer = VectorQuantizer(
            config=self.config,
            n_head=QKVGProducer.get_n_kv(self.config),
        )
        self.xl_bias_producer = XLBiasProducer(self.config)
        proj_kwargs = dict(
            kernel_init=self.w_init,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res_proj = nn.Dense(self.d_model, **proj_kwargs)
        self.dropres = nn.Dropout(
            self.p_dropres, rng_collection="ephemeral", deterministic=not self.is_train
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def pre_reshape(x, config):
        # call this just once, at start of transformer layer stack
        dims = chex.Dimensions(
            B=x.shape[0],
            T=config.sequence_len,
            R=config.sequence_len // config.block_len,
            C=config.block_len,
            D=config.d_model,
        )
        chex.assert_shape(x, dims["BTD"])
        return jnp.reshape(x, dims["BRCD"])

    @staticmethod
    def post_reshape(x, config):
        # call this just once, at end of transformer layer stack
        dims = chex.Dimensions(
            B=x.shape[0],
            T=config.sequence_len,
            R=config.sequence_len // config.block_len,
            C=config.block_len,
            D=config.d_model,
        )
        chex.assert_shape(x, dims["BRCD"])
        return jnp.reshape(x, dims["BTD"])

    def compute_cache_vars(self, z: jax.Array, v: jax.Array, dims: chex.Dimensions):
        # this function computes our per-block cache variables parallel using matmul,
        # which is similar to hua et al., 2022.
        delta = jax.nn.one_hot(z, num_classes=self.n_code, dtype=self.dtype, axis=-1)
        chex.assert_shape(delta, dims["BhRCS"])
        delta1_by_block = jnp.einsum("bhrcs->bhrs", delta)
        delta1_cumul_by_block = jnp.cumsum(delta1_by_block, axis=2)
        delta1_fracs_by_block = jnp.einsum(
            "bhrs,bhgs->bhsrg",
            jnp.reciprocal(jnp.clip(delta1_cumul_by_block, a_min=1.0)),
            delta1_by_block,
        )  # safe divide here has no impact on result, since zero denom implies zero row
        delta1_fracs_by_block = jnp.tril(delta1_fracs_by_block)
        cache_var_lower = jnp.pad(
            delta1_cumul_by_block[:, :, :-2], ((0, 0), (0, 0), (2, 0), (0, 0))
        )

        deltav_by_block = jnp.einsum("bhrcs,bhrcv->bhrsv", delta, v)
        deltav_by_block_normalized = deltav_by_block / jnp.clip(
            delta1_by_block[..., None], a_min=1.0
        )  # safe divide here has no impact on result, since zero denom implies zero row
        deltav_by_block_cumul_normalized = jnp.einsum(
            "bhsrg,bhgsv->bhrsv", delta1_fracs_by_block, deltav_by_block_normalized
        )
        cache_var_upper_div_lower = jnp.pad(
            deltav_by_block_cumul_normalized[:, :, :-2],
            ((0, 0), (0, 0), (2, 0), (0, 0), (0, 0)),
        )
        chex.assert_shape(cache_var_lower, dims["BhRS"])
        chex.assert_shape(cache_var_upper_div_lower, dims["BhRSV"])
        chex.assert_trees_all_equal_dtypes(
            delta, cache_var_lower, cache_var_upper_div_lower, v
        )
        return cache_var_lower, cache_var_upper_div_lower

    def __call__(self, x):
        dims = chex.Dimensions(
            B=x.shape[0],
            T=self.sequence_len,
            R=self.sequence_len // self.block_len,
            C=self.block_len,
            W=2 * self.block_len,
            D=self.d_model,
            H=QKVGProducer.get_n_q(self.config),
            h=QKVGProducer.get_n_kv(self.config),
            S=self.n_code,
            K=self.d_k,
            V=QKVGProducer.get_d_v(self.config),
            O=QKVGProducer.get_n_q(self.config) * QKVGProducer.get_d_v(self.config),
            i=1,
        )
        x_tilde = self.input_ln(x)
        q = self.qkvg_producer.get_queries(x_tilde=x_tilde)
        k, v = self.qkvg_producer.get_keys_and_values(x_tilde=x_tilde)
        q, k, v = map(lambda y: jnp.transpose(y, (0, 3, 1, 2, 4)), [q, k, v])
        chex.assert_shape(q, dims["BHRCK"])
        chex.assert_shape(k, dims["BhRCK"])
        chex.assert_shape(v, dims["BhRCV"])
        chex.assert_trees_all_equal_dtypes(q, k, v)

        vq_out = self.quantizer(k)
        z = vq_out["shortcodes"]
        k_hat = vq_out["quantized"]
        l_commit = vq_out["l_commit"]
        l_codebook = vq_out["l_codebook"]
        chex.assert_shape(z, dims["BhRC"])
        chex.assert_shape(k_hat, dims["BhRCK"])
        chex.assert_shape(l_commit, [])
        chex.assert_shape(l_codebook, [])
        chex.assert_trees_all_equal_dtypes(k, k_hat)

        xl_r, xl_u, xl_v = self.xl_bias_producer(k_len=2 * self.block_len)
        xl_r, xl_u, xl_v = map(lambda y: jnp.expand_dims(y, -3), [xl_r, xl_u, xl_v])
        chex.assert_shape(xl_r, dims["hiWK"])
        chex.assert_shape(xl_u, dims["iHiiK"])
        chex.assert_shape(xl_v, dims["iHiiK"])
        chex.assert_trees_all_equal_dtypes(xl_r, xl_u, xl_v)
        q_plus_xlu = q + xl_u
        q_plus_xlv = q + xl_v
        chex.assert_shape(q_plus_xlu, dims["BHRCK"])
        chex.assert_shape(q_plus_xlv, dims["BHRCK"])
        chex.assert_trees_all_equal_dtypes(q, q_plus_xlu, q_plus_xlv)

        k_hat_prev = jnp.pad(k_hat[:, :, :-1], ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
        v_prev = jnp.pad(v[:, :, :-1], ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
        codebook = self.quantizer.get_codebook()
        chex.assert_shape(k_hat_prev, dims["BhRCK"])
        chex.assert_shape(v_prev, dims["BhRCV"])
        chex.assert_shape(codebook, dims["hSK"])
        chex.assert_trees_all_equal_dtypes(k_hat_prev, v_prev, codebook)

        scores_ac_present = jnp.einsum("bhrik,bhrjk->bhrij", q_plus_xlu, k_hat)
        scores_ac_prev = jnp.einsum("bhrik,bhrjk->bhrij", q_plus_xlu, k_hat_prev)
        scores_cache = jnp.einsum("bhrck,hsk->bhrcs", q_plus_xlu, codebook)
        chex.assert_shape(scores_ac_present, dims["BHRCC"])
        chex.assert_shape(scores_ac_prev, dims["BHRCC"])
        chex.assert_shape(scores_cache, dims["BHRCS"])
        chex.assert_trees_all_equal_dtypes(
            scores_ac_present, scores_ac_prev, scores_cache
        )

        scores_bd_recent = jnp.einsum("bhrck,hrwk->bhrcw", q_plus_xlv, xl_r)
        scores_bd_recent = XLBiasProducer.rel_shift(scores_bd_recent)
        scores_bd_recent *= XLBiasProducer.get_causal_keep_mask(
            q_len=self.block_len,
            k_len=2 * self.block_len,
            with_locality=True,  # zero out the biases left of the sliding window
        )[None, None, None, ...]
        scores_bd_prev, scores_bd_present = jnp.split(scores_bd_recent, 2, axis=-1)
        chex.assert_shape(scores_bd_present, dims["BHRCC"])
        chex.assert_shape(scores_bd_prev, dims["BHRCC"])
        chex.assert_trees_all_equal_dtypes(scores_bd_present, scores_bd_prev)

        scores_present = scores_ac_present + scores_bd_present
        scores_prev = scores_ac_prev + scores_bd_prev
        chex.assert_shape(scores_present, dims["BHRCC"])
        chex.assert_shape(scores_prev, dims["BHRCC"])
        chex.assert_trees_all_equal_dtypes(scores_present, scores_prev)

        # mask out non-causal connections within each block
        mask_present = XLBiasProducer.get_causal_keep_mask(
            q_len=self.block_len,
            k_len=self.block_len,
            with_locality=False,
        )[None, None, None, ...]
        scores_present -= jnp.array(INFTY_APPROX, dtype=self.dtype) * jnp.logical_not(
            mask_present
        )
        # mask out attn weight contrib from zero padding in k_prev
        scores_prev = jnp.pad(
            scores_prev[:, :, 1:],
            ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)),
            constant_values=-INFTY_APPROX,
        )
        chex.assert_shape(scores_present, dims["BHRCC"])
        chex.assert_trees_all_equal_dtypes(scores_present, scores_prev, scores_cache)

        cache_var_lower, cache_var_upper_div_lower = self.compute_cache_vars(z, v, dims)
        count_biases = jnp.where(
            jnp.equal(cache_var_lower, jnp.zeros_like(cache_var_lower)),
            -INFTY_APPROX,
            jnp.log(jnp.maximum(cache_var_lower, jnp.ones_like(cache_var_lower))),
        )
        scores_cache += jnp.expand_dims(count_biases, axis=-2)

        # subtract max score for stability (ok due to softmax shift-invariance)
        scores_present_max = jnp.max(scores_present, axis=-1)
        scores_prev_max = jnp.max(scores_prev, axis=-1)
        scores_cache_max = jnp.max(scores_cache, axis=-1)
        chex.assert_shape(scores_present_max, dims["BHRC"])
        chex.assert_shape(scores_prev_max, dims["BHRC"])
        chex.assert_shape(scores_cache_max, dims["BHRC"])
        max_scores = jnp.maximum(
            jnp.maximum(scores_present_max, scores_prev_max), scores_cache_max
        )
        max_scores = jax.lax.stop_gradient(max_scores)
        scores_present -= max_scores[..., None]
        scores_prev -= max_scores[..., None]
        scores_cache -= max_scores[..., None]

        # exponentiate the attention scores
        a_present = jnp.exp(scores_present)
        a_prev = jnp.exp(scores_prev)
        a_cache = jnp.exp(scores_cache)
        chex.assert_shape(a_present, dims["BHRCC"])
        chex.assert_shape(a_prev, dims["BHRCC"])
        chex.assert_shape(a_cache, dims["BHRCS"])
        chex.assert_trees_all_equal_dtypes(a_present, a_prev, a_cache)

        # compute attn normalizer, attn weights, and attn outputs
        d = jnp.sum(a_present, axis=-1)
        d += jnp.sum(a_prev, axis=-1)
        d += jnp.sum(a_cache, axis=-1)
        w_present = a_present / d[..., None]
        w_prev = a_prev / d[..., None]
        w_cache = a_cache / d[..., None]
        wv = jnp.einsum("bhrij,bhrjv->brihv", w_present, v)
        wv += jnp.einsum("bhrij,bhrjv->brihv", w_prev, v_prev)
        wv += jnp.einsum("bhris,bhrsv->brihv", w_cache, cache_var_upper_div_lower)

        wv = jnp.reshape(wv, dims["BRCO"])
        if self.head_type == "shga":
            wv *= self.qkvg_producer.get_gates(x_tilde=x_tilde)
        res = self.res_proj(wv)
        res = self.dropres(res)
        chex.assert_trees_all_equal_dtypes(res, wv, d, q, k, v, x, x_tilde)
        return dict(
            res=res,
            l_commit=l_commit,
            l_codebook=l_codebook,
        )


class FullAttention(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        assert self.sequence_len // self.block_len > 2
        self.tau = self.d_k**0.5
        self.input_ln = RMSNorm()
        self.qkvg_producer = QKVGProducer(self.config)
        self.xl_bias_producer = XLBiasProducer(self.config)
        proj_kwargs = dict(
            kernel_init=self.w_init,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res_proj = nn.Dense(self.d_model, **proj_kwargs)
        self.dropres = nn.Dropout(
            self.p_dropres, rng_collection="ephemeral", deterministic=not self.is_train
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def pre_reshape(x, config):
        # call this just once, at start of transformer layer stack
        dims = chex.Dimensions(
            B=x.shape[0],
            T=config.sequence_len,
            R=config.sequence_len // config.block_len,
            C=config.block_len,
            D=config.d_model,
        )
        chex.assert_shape(x, dims["BTD"])
        return x

    @staticmethod
    def post_reshape(x, config):
        # call this just once, at end of transformer layer stack
        dims = chex.Dimensions(
            B=x.shape[0],
            T=config.sequence_len,
            R=config.sequence_len // config.block_len,
            C=config.block_len,
            D=config.d_model,
        )
        chex.assert_shape(x, dims["BTD"])
        return x

    def __call__(self, x):
        dims = chex.Dimensions(
            B=x.shape[0],
            T=self.sequence_len,
            D=self.d_model,
            H=QKVGProducer.get_n_q(self.config),
            h=QKVGProducer.get_n_kv(self.config),
            K=self.d_k,
            V=QKVGProducer.get_d_v(self.config),
            O=QKVGProducer.get_n_q(self.config) * QKVGProducer.get_d_v(self.config),
            i=1,
        )
        x_tilde = self.input_ln(x)
        q = self.qkvg_producer.get_queries(x_tilde=x_tilde)
        k, v = self.qkvg_producer.get_keys_and_values(x_tilde=x_tilde)
        q, k, v = map(lambda y: jnp.transpose(y, (0, 2, 1, 3)), [q, k, v])
        chex.assert_shape(q, dims["BHTK"])
        chex.assert_shape(k, dims["BhTK"])
        chex.assert_shape(v, dims["BhTV"])
        chex.assert_trees_all_equal_dtypes(q, k, v)

        xl_r, xl_u, xl_v = self.xl_bias_producer(k_len=self.sequence_len)
        chex.assert_shape(xl_r, dims["hTK"])
        chex.assert_shape(xl_u, dims["iHiK"])
        chex.assert_shape(xl_v, dims["iHiK"])
        chex.assert_trees_all_equal_dtypes(xl_r, xl_u, xl_v)
        q_plus_xlu = q + xl_u
        q_plus_xlv = q + xl_v
        chex.assert_shape(q_plus_xlu, dims["BHTK"])
        chex.assert_shape(q_plus_xlv, dims["BHTK"])
        chex.assert_trees_all_equal_dtypes(q, q_plus_xlu, q_plus_xlv)

        scores_ac = jnp.einsum("bhik,bhjk->bhij", q_plus_xlu, k)
        scores_bd = jnp.einsum("bhik,hjk->bhij", q_plus_xlv, xl_r)
        scores_bd = XLBiasProducer.rel_shift(scores_bd)
        chex.assert_shape(scores_ac, dims["BHTT"])
        chex.assert_shape(scores_bd, dims["BHTT"])
        chex.assert_trees_all_equal_dtypes(scores_ac, scores_bd)

        scores = scores_ac + scores_bd
        mask = XLBiasProducer.get_causal_keep_mask(
            q_len=self.sequence_len,
            k_len=self.sequence_len,
            with_locality=False,
        )[None, None, ...]
        scores -= jnp.array(INFTY_APPROX, dtype=self.dtype) * jnp.logical_not(mask)
        chex.assert_shape(scores, dims["BHTT"])
        chex.assert_shape(mask, dims["iiTT"])

        w = jax.nn.softmax(scores, axis=-1)
        wv = jnp.einsum("bhij,bhjv->bihv", w, v)
        chex.assert_shape(w, dims["BHTT"])
        chex.assert_shape(wv, dims["BTHV"])

        wv = jnp.reshape(wv, dims["BTO"])
        if self.head_type == "shga":
            wv *= self.qkvg_producer.get_gates(x_tilde=x_tilde)
        res = self.res_proj(wv)
        res = self.dropres(res)
        chex.assert_trees_all_equal_dtypes(res, wv, q, k, v, x, x_tilde)
        return dict(
            res=res,
            l_commit=jnp.zeros([], dtype=self.dtype),
            l_codebook=jnp.zeros([], dtype=self.dtype),
        )


def get_attn_cls(attn_type: str):
    if attn_type == "vq":
        return VQAttention
    if attn_type == "full":
        return FullAttention
    if attn_type.endswith("_old"):
        return ScannedAttnOld
    if attn_type.endswith("_unwrapped") and "_old" in attn_type:
        return get_attn_cls_old(attn_type)
    raise NotImplementedError(f"Unknown attention type {attn_type}")
