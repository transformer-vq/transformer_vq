import dataclasses

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_vq.nn.norm import RMSNorm
from transformer_vq.nn.pe import XLBiasProducer
from transformer_vq.nn.qkv import QKVGProducer
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import VectorQuantizer

INFTY_APPROX = 1e30  # large value approximating infinity


class VQAttentionOld(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
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
    def initial_state(config, batch_size):
        n_kv_head = QKVGProducer.get_n_kv(config)
        prefix = [batch_size, n_kv_head]
        s = config.n_code
        m = config.block_len
        d_k = config.d_k
        d_v = QKVGProducer.get_d_v(config)
        return dict(
            pos_offset=jnp.array(0, dtype=jnp.int32),
            xlcache=dict(
                z=jnp.full(shape=[*prefix, m], dtype=jnp.int32, fill_value=s),
                k_hat=jnp.zeros([*prefix, m, d_k], dtype=config.dtype),
                v=jnp.zeros([*prefix, m, d_v], dtype=config.dtype),
            ),
            aggcache=dict(
                upper_div_lower=jnp.zeros([*prefix, s, d_v], dtype=config.dtype),
                lower=jnp.zeros([*prefix, s], dtype=config.dtype),
            ),
        )

    @staticmethod
    def get_agg_biases(lower):
        result = jnp.where(
            jnp.equal(lower, jnp.zeros_like(lower)),
            -INFTY_APPROX,
            jnp.log(jnp.maximum(lower, jnp.ones_like(lower))),  # this is never nan
        )
        return result

    def attn(self, present_q, present_k, present_v, state):
        bsz = present_q.shape[0]
        dims = chex.Dimensions(
            B=bsz,
            L=self.block_len,
            M=self.block_len,
            W=2 * self.block_len,
            H=QKVGProducer.get_n_q(self.config),
            h=QKVGProducer.get_n_kv(self.config),
            S=self.n_code,
            K=self.d_k,
            V=QKVGProducer.get_d_v(self.config),
            i=1,
        )
        chex.assert_trees_all_equal_dtypes(
            present_v,
            state["xlcache"]["v"],
            state["aggcache"]["upper_div_lower"],
            state["aggcache"]["lower"],
        )
        chex.assert_shape(present_q, dims["BHLK"])
        chex.assert_shape(present_k, dims["BhLK"])
        chex.assert_shape(present_v, dims["BhLV"])

        # quantize keys, get commit loss and codebook surrogate loss
        vq_output_dict = self.quantizer(jnp.expand_dims(present_k, 2))
        present_z = jnp.squeeze(vq_output_dict["shortcodes"], 2)
        present_k_hat = jnp.squeeze(vq_output_dict["quantized"], 2)
        l_commit = vq_output_dict["l_commit"]
        l_codebook = vq_output_dict["l_codebook"]
        chex.assert_shape(present_z, dims["BhL"])
        chex.assert_shape(present_k_hat, dims["BhLK"])
        chex.assert_trees_all_equal_dtypes(present_k, present_k_hat)

        # concatenate sliding window cache k/v onto current block
        xlcache = state["xlcache"]
        aggcache = state["aggcache"]
        chex.assert_shape(xlcache["z"], dims["BhM"])
        chex.assert_shape(xlcache["k_hat"], dims["BhMK"])
        chex.assert_shape(xlcache["v"], dims["BhMV"])
        recent_z = jnp.concatenate([xlcache["z"], present_z], axis=-1)
        recent_k_hat = jnp.concatenate([xlcache["k_hat"], present_k_hat], axis=-2)
        recent_v = jnp.concatenate([xlcache["v"], present_v], axis=-2)
        chex.assert_shape(recent_z, dims["BhW"])
        chex.assert_shape(recent_k_hat, dims["BhWK"])
        chex.assert_shape(recent_v, dims["BhWV"])

        # compute xl bias helpers
        xl_r, xl_u, xl_v = self.xl_bias_producer(k_len=2 * self.block_len)

        # compute aggcache scores
        c = self.quantizer.get_codebook()
        cache_scores = jnp.einsum("bhlk,hsk->bhls", present_q + xl_u, c)
        cache_biases = VQAttentionOld.get_agg_biases(aggcache["lower"])
        cache_biases = jnp.expand_dims(cache_biases, axis=-2)
        cache_scores += cache_biases

        # compute recent scores (present and xlcache)
        recent_scores_ac = jnp.einsum("bhlk,bhwk->bhlw", present_q + xl_u, recent_k_hat)
        recent_scores_bd = jnp.einsum("bhlk,hwk->bhlw", present_q + xl_v, xl_r)
        recent_scores_bd = XLBiasProducer.rel_shift(recent_scores_bd)
        recent_scores_bd *= XLBiasProducer.get_causal_keep_mask(
            q_len=self.block_len,
            k_len=2 * self.block_len,
            invalid_len=jax.nn.relu(self.block_len - state["pos_offset"]),
            with_alloc=True,
            with_locality=True,  # zero out the biases left of the sliding window
        )[None, None, ...]
        recent_scores = recent_scores_ac + recent_scores_bd
        keep_mask = XLBiasProducer.get_causal_keep_mask(
            q_len=self.block_len,
            k_len=2 * self.block_len,
            invalid_len=jax.nn.relu(self.block_len - state["pos_offset"]),
            with_alloc=True,
            with_locality=False,  # allow attending to bubble left of the sliding window
        )[None, None, ...]
        recent_scores -= jnp.array(INFTY_APPROX, dtype=self.dtype) * jnp.logical_not(
            keep_mask
        )

        # subtract max score for stability (ok due to softmax shift-invariance)
        cache_max_scores = jnp.max(cache_scores, axis=-1)  # BHL
        recent_max_scores = jnp.max(recent_scores, axis=-1)  # BHL
        max_scores = jnp.maximum(cache_max_scores, recent_max_scores)  # BHL
        max_scores = jax.lax.stop_gradient(max_scores)
        chex.assert_shape(max_scores, dims["BHL"])
        cache_scores -= max_scores[..., None]
        recent_scores -= max_scores[..., None]
        cache_a = jnp.exp(cache_scores)
        recent_a = jnp.exp(recent_scores)
        chex.assert_shape(cache_a, dims["BHLS"])
        chex.assert_shape(recent_a, dims["BHLW"])

        # compute per-query normalizer d and divide unnormalized weights a by it first,
        # so the numerically unstable expression av is never materialized.
        d = jnp.sum(recent_a, axis=-1)
        d += jnp.sum(cache_a, axis=-1)
        wv = jnp.einsum("bhlw,bhwv->blhv", recent_a / d[..., None], recent_v)
        wv += jnp.einsum(
            "bhls,bhsv->blhv", cache_a / d[..., None], aggcache["upper_div_lower"]
        )
        chex.assert_shape(wv, dims["BLHV"])
        wv = jnp.reshape(wv, [bsz, self.block_len, -1])
        return dict(
            attn_out=wv,
            recent_z=recent_z,
            recent_k_hat=recent_k_hat,
            recent_v=recent_v,
            l_commit=l_commit,
            l_codebook=l_codebook,
        )

    def update_state(self, recent_z, recent_k_hat, recent_v, state):
        bsz, *_ = recent_z.shape
        dims = chex.Dimensions(
            B=bsz,
            L=self.block_len,
            M=self.block_len,
            S=self.n_code,
            H=QKVGProducer.get_n_q(self.config),
            h=QKVGProducer.get_n_kv(self.config),
            K=self.d_k,
            V=QKVGProducer.get_d_v(self.config),
        )
        aggcache = state["aggcache"]
        chex.assert_shape(aggcache["upper_div_lower"], dims["BhSV"])
        chex.assert_shape(aggcache["lower"], dims["BhS"])
        chex.assert_shape(recent_z[..., : -self.block_len], dims["BhL"])
        chex.assert_shape(recent_v[..., : -self.block_len, :], dims["BhLV"])
        chex.assert_shape(recent_k_hat[..., : -self.block_len, :], dims["BhLK"])
        chex.assert_shape(recent_z[..., -self.block_len :], dims["BhM"])
        chex.assert_shape(recent_v[..., -self.block_len :, :], dims["BhMV"])
        chex.assert_shape(recent_k_hat[..., -self.block_len :, :], dims["BhMK"])
        # compute kronecker deltas; invalid z's from xlcache init encode to zero vecs
        delta = jax.nn.one_hot(
            recent_z[..., : -self.block_len],
            num_classes=self.n_code,
            dtype=self.dtype,
            axis=-1,
        )  # BHLS
        # compute new position offset
        new_pos_offset = state["pos_offset"] + self.block_len
        new_lower = jnp.add(aggcache["lower"], jnp.sum(delta, axis=-2))
        # compute updated upper cache variable (stored in relative format for stability)
        # i.e., we compute new_upper_div_lower by dividing axis S by counts in new_lower
        f1 = aggcache["lower"] / jnp.clip(new_lower, a_min=1.0)
        f2 = delta / jnp.expand_dims(jnp.clip(new_lower, a_min=1.0), -2)
        new_upper_div_lower = jnp.add(
            f1[..., None] * aggcache["upper_div_lower"],
            jnp.einsum("bhls,bhlv->bhsv", f2, recent_v[..., : -self.block_len, :]),
        )
        new_state = dict(
            pos_offset=new_pos_offset,
            xlcache=dict(
                z=recent_z[..., -self.block_len :],
                k_hat=recent_k_hat[..., -self.block_len :, :],
                v=recent_v[..., -self.block_len :, :],
            ),
            aggcache=dict(
                lower=new_lower,
                upper_div_lower=new_upper_div_lower,
            ),
        )
        return new_state

    def __call__(self, state, x):
        x_tilde = self.input_ln(x)
        q = self.qkvg_producer.get_queries(x_tilde=x_tilde)
        k, v = self.qkvg_producer.get_keys_and_values(x_tilde=x_tilde)
        q, k, v = map(lambda y: jnp.transpose(y, (0, 2, 1, 3)), [q, k, v])
        attn_output_dict = self.attn(q, k, v, state)
        wv = attn_output_dict.get("attn_out")
        if self.head_type == "shga":
            wv *= self.qkvg_producer.get_gates(x_tilde=x_tilde)
        res = self.res_proj(wv)
        res = self.dropres(res)
        chex.assert_trees_all_equal_dtypes(x, res)
        new_state = self.update_state(
            recent_z=attn_output_dict.get("recent_z"),
            recent_k_hat=attn_output_dict.get("recent_k_hat"),
            recent_v=attn_output_dict.get("recent_v"),
            state=state,
        )
        output_dict = dict(
            res=res,
            l_commit=attn_output_dict.get("l_commit")[..., None],
            l_codebook=attn_output_dict.get("l_codebook")[..., None],
        )
        return new_state, output_dict


class FullAttentionOld(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
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
    def initial_state(config, batch_size):
        n_kv_head = QKVGProducer.get_n_kv(config)
        prefix = [batch_size, n_kv_head]
        m = config.sequence_len
        d_k = config.d_k
        d_v = QKVGProducer.get_d_v(config)
        return dict(
            pos_offset=jnp.array(0, dtype=jnp.int32),
            xlcache=dict(
                k=jnp.zeros([*prefix, m, d_k], dtype=config.dtype),
                v=jnp.zeros([*prefix, m, d_v], dtype=config.dtype),
            ),
        )

    def attn(self, present_q, present_k, present_v, state):
        bsz = present_q.shape[0]
        dims = chex.Dimensions(
            B=bsz,
            L=self.block_len,
            M=self.sequence_len,
            W=self.sequence_len + self.block_len,
            H=QKVGProducer.get_n_q(self.config),
            h=QKVGProducer.get_n_kv(self.config),
            S=self.n_code,
            K=self.d_k,
            V=QKVGProducer.get_d_v(self.config),
            i=1,
        )
        chex.assert_trees_all_equal_dtypes(present_v, state["xlcache"]["v"])
        chex.assert_shape(present_q, dims["BHLK"])
        chex.assert_shape(present_k, dims["BhLK"])
        chex.assert_shape(present_v, dims["BhLV"])

        # concatenate sliding window cache k/v onto current block
        xlcache = state["xlcache"]
        chex.assert_shape(xlcache["k"], dims["BhMK"])
        chex.assert_shape(xlcache["v"], dims["BhMV"])
        recent_k = jnp.concatenate([xlcache["k"], present_k], axis=-2)
        recent_v = jnp.concatenate([xlcache["v"], present_v], axis=-2)
        chex.assert_shape(recent_k, dims["BhWK"])
        chex.assert_shape(recent_v, dims["BhWV"])

        # compute xl bias helpers
        xl_r, xl_u, xl_v = self.xl_bias_producer(
            k_len=self.sequence_len + self.block_len
        )

        # compute recent scores (present and xlcache)
        recent_scores_ac = jnp.einsum("bhlk,bhwk->bhlw", present_q + xl_u, recent_k)
        recent_scores_bd = jnp.einsum("bhlk,hwk->bhlw", present_q + xl_v, xl_r)
        recent_scores_bd = XLBiasProducer.rel_shift(recent_scores_bd)
        recent_scores = recent_scores_ac + recent_scores_bd
        keep_mask = XLBiasProducer.get_causal_keep_mask(
            q_len=self.block_len,
            k_len=self.sequence_len + self.block_len,
            invalid_len=jax.nn.relu(self.sequence_len - state["pos_offset"]),
            with_alloc=True,
            with_locality=True,
        )[None, None, ...]
        recent_scores -= jnp.array(INFTY_APPROX, dtype=self.dtype) * jnp.logical_not(
            keep_mask
        )

        # subtract max score for stability (ok due to softmax shift-invariance)
        recent_max_scores = jnp.max(recent_scores, axis=-1)  # BHL
        max_scores = jax.lax.stop_gradient(recent_max_scores)
        recent_scores -= max_scores[..., None]
        recent_a = jnp.exp(recent_scores)
        chex.assert_shape(recent_a, dims["BHLW"])

        # compute per-query normalizer d and divide unnormalized weights a by it first,
        # so the numerically unstable expression av is never materialized.
        d = jnp.sum(recent_a, axis=-1)
        wv = jnp.einsum("bhlw,bhwv->blhv", recent_a / d[..., None], recent_v)
        chex.assert_shape(wv, dims["BLHV"])
        wv = jnp.reshape(wv, [bsz, self.block_len, -1])
        return dict(
            attn_out=wv,
            recent_k=recent_k,
            recent_v=recent_v,
        )

    def update_state(self, recent_k, recent_v, state):
        bsz, *_ = recent_k.shape
        dims = chex.Dimensions(
            B=bsz,
            L=self.block_len,
            M=self.sequence_len,
            S=self.n_code,
            H=QKVGProducer.get_n_q(self.config),
            h=QKVGProducer.get_n_kv(self.config),
            K=self.d_k,
            V=QKVGProducer.get_d_v(self.config),
        )
        chex.assert_shape(recent_k[..., : -self.sequence_len, :], dims["BhLK"])
        chex.assert_shape(recent_v[..., : -self.sequence_len, :], dims["BhLV"])
        new_state = dict(
            pos_offset=state["pos_offset"] + self.block_len,
            xlcache=dict(
                k=recent_k[..., -self.sequence_len :, :],
                v=recent_v[..., -self.sequence_len :, :],
            ),
        )
        return new_state

    def __call__(self, state, x):
        x_tilde = self.input_ln(x)
        q = self.qkvg_producer.get_queries(x_tilde=x_tilde)
        k, v = self.qkvg_producer.get_keys_and_values(x_tilde=x_tilde)
        q, k, v = map(lambda y: jnp.transpose(y, (0, 2, 1, 3)), [q, k, v])
        attn_output_dict = self.attn(q, k, v, state)
        wv = attn_output_dict.get("attn_out")
        if self.head_type == "shga":
            wv *= self.qkvg_producer.get_gates(x_tilde=x_tilde)
        res = self.res_proj(wv)
        res = self.dropres(res)
        chex.assert_trees_all_equal_dtypes(x, res)
        new_state = self.update_state(
            recent_k=attn_output_dict.get("recent_k"),
            recent_v=attn_output_dict.get("recent_v"),
            state=state,
        )
        output_dict = dict(
            res=res,
            l_commit=jnp.zeros([], dtype=self.dtype)[..., None],
            l_codebook=jnp.zeros([], dtype=self.dtype)[..., None],
        )
        return new_state, output_dict


def get_attn_cls_old(attn_type: str):
    if attn_type == "vq_old_unwrapped":
        return VQAttentionOld
    if attn_type == "full_old_unwrapped":
        return FullAttentionOld
    raise NotImplementedError(f"Unknown old attention type {attn_type}")


class ScannedAttnOld(nn.Module):
    config: TransformerConfig

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

    @nn.compact
    def __call__(self, x):
        dims = chex.Dimensions(
            B=x.shape[0],
            T=self.config.sequence_len,
            R=self.config.sequence_len // self.config.block_len,
            C=self.config.block_len,
            D=self.config.d_model,
        )
        chex.assert_shape(x, dims["BRCD"])
        attn_cls = get_attn_cls_old(self.config.attn_type + "_unwrapped")
        attn_state_init = attn_cls.initial_state(self.config, x.shape[0])
        attn_scan_args = dict(
            variable_broadcast="params",
            split_rngs=dict(
                params=False,
                timeless=False,
                ephemeral=True,
            ),
            in_axes=1,
            out_axes=1,
        )
        _, out = nn.scan(attn_cls, **attn_scan_args)(self.config)(attn_state_init, x)
        return dict(
            res=out["res"],
            l_commit=jnp.mean(out["l_commit"]),
            l_codebook=jnp.mean(out["l_codebook"]),
        )
