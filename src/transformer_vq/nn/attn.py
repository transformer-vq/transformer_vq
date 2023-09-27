import dataclasses

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from transformer_vq.nn.grad import sg
from transformer_vq.nn.norm import LayerNorm
from transformer_vq.nn.pe import get_sinusoid_embs
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import LearnableVQ

MASK_INFTY_APPROX = 1e30  # mask value approximating infinity


class VQAttention(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        self.tau = self.d_k**0.5
        self.input_ln = LayerNorm(self.d_model, self.param_dtype)
        q_ch = self.n_head * self.d_k
        k_ch = self.n_head * self.d_k
        v_ch = self.n_head * self.d_v
        self.q_ln = LayerNorm(self.d_k, self.param_dtype, gain=False, bias=False)
        self.k_ln = LayerNorm(self.d_k, self.param_dtype, gain=False, bias=False)
        proj_kwargs = dict(
            kernel_init=self.w_init,
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.q_proj = nn.Dense(q_ch, **proj_kwargs)
        self.kvg_proj = nn.Dense(k_ch + v_ch + v_ch, **proj_kwargs)
        self.r_proj = nn.Dense(k_ch, **proj_kwargs)
        proj_kwargs.update(dict(kernel_init=self.r_init))
        self.res_proj = nn.Dense(self.d_model, **proj_kwargs)
        self.xl_u = self.param("u", self.b_init, [q_ch], self.param_dtype)
        self.xl_v = self.param("v", self.b_init, [q_ch], self.param_dtype)
        self.quantizer = LearnableVQ(self.config)
        self.dropsin = nn.Dropout(
            self.p_dropsin, rng_collection="timeless", deterministic=not self.is_train
        )
        self.dropres = nn.Dropout(
            self.p_dropres, rng_collection="ephemeral", deterministic=not self.is_train
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def initial_state(config, batch_size):
        prefix = [batch_size, config.n_head]
        s = config.n_code
        m = config.mem_len
        d_k = config.d_k
        d_v = config.d_v
        return dict(
            pos_offset=jnp.array(0, dtype=jnp.int32),
            xlcache=dict(
                z=jnp.full(shape=[*prefix, m], dtype=jnp.int32, fill_value=s),
                k_hat=jnp.zeros([*prefix, m, d_k], dtype=config.param_dtype),
                v=jnp.zeros([*prefix, m, d_v], dtype=config.dtype),
                doc_ids=jnp.zeros([batch_size, m], jnp.int32),
            ),
            aggcache=dict(
                upper_div_lower=jnp.zeros([*prefix, s, d_v], dtype=config.dtype),
                lower=jnp.zeros([*prefix, s], dtype=config.dtype),
                latest_doc_id=jnp.zeros([batch_size], jnp.int32),
            ),
        )

    @staticmethod
    def rel_shift(x):
        *leading_shape, present_len, past_len = x.shape
        pad_spec = [(0, 0)] * len(leading_shape) + [(0, 0), (1, 0)]
        x = jnp.pad(x, pad_spec)
        x = jnp.reshape(x, [*leading_shape, past_len + 1, present_len])
        x = x[..., 1:, :]
        x = jnp.reshape(x, [*leading_shape, present_len, past_len])
        return x

    @staticmethod
    def get_causal_mask(block_len, mem_len, invalid_len, with_locality):
        # invalid len must be a jax array to work properly with jit
        chex.assert_shape(invalid_len, [])
        assert block_len > 0 and mem_len >= 0
        i = jnp.arange(block_len)[..., None]
        j = jnp.arange(mem_len + block_len)[None, ...]
        alloc_mask = jnp.greater_equal(j, jnp.array([invalid_len])[None, ...])
        causal_mask = jnp.less_equal(j - mem_len, i)
        window_mask = jnp.greater_equal(j, i)
        keep_mask = jnp.logical_and(alloc_mask, causal_mask)
        if with_locality:
            keep_mask = jnp.logical_and(keep_mask, window_mask)
        return keep_mask

    @staticmethod
    def get_agg_biases(lower):
        result = jnp.where(
            jnp.equal(lower, jnp.zeros_like(lower)),
            -MASK_INFTY_APPROX,
            jnp.log(jnp.maximum(lower, jnp.ones_like(lower))),  # this is never nan
        )
        return result

    def get_q(self, x_tilde):
        bsz, present_len, *_ = x_tilde.shape
        q = self.q_proj(x_tilde)
        q = jnp.reshape(q, [bsz, present_len, self.n_head, self.d_k])
        q = self.q_ln(q) * (self.tau**-0.5)
        q = jnp.transpose(q, (0, 2, 1, 3))
        return q.astype(self.param_dtype)

    def get_kvg(self, x_tilde):
        bsz, present_len, *_ = x_tilde.shape
        kvg = self.kvg_proj(x_tilde)
        inds = np.cumsum(np.array([self.d_k, self.d_v]))
        k, v, g = jnp.split(kvg, self.n_head * inds, axis=-1)
        chex.assert_shape(k, [bsz, present_len, self.n_head * self.d_k])
        chex.assert_shape(v, [bsz, present_len, self.n_head * self.d_v])
        chex.assert_shape(g, [bsz, present_len, self.n_head * self.d_v])
        k = jnp.reshape(k, [bsz, present_len, self.n_head, self.d_k])
        v = jnp.reshape(v, [bsz, present_len, self.n_head, self.d_v])
        k = self.k_ln(k) * (self.tau**-0.5)
        v = jax.nn.silu(v)
        g = jax.nn.silu(g)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        return k.astype(self.param_dtype), v, g

    def get_xl_helpers(self):
        # compute helpers for xl biases (z dai et al., 2019)
        xl_r = get_sinusoid_embs(
            length=self.mem_len + self.block_len,
            width=self.d_model,
            lam=self.pe_lam,
            flip=True,
        )
        xl_r = self.dropsin(xl_r)
        xl_r = self.r_proj(xl_r)
        xl_r = jnp.reshape(xl_r, [self.mem_len + self.block_len, self.n_head, self.d_k])
        xl_r = jnp.transpose(xl_r, (1, 0, 2))
        xl_r = xl_r.astype(self.param_dtype) * (self.tau**-0.5)
        xl_u = jnp.reshape(self.xl_u, [1, self.n_head, 1, self.d_k]) * (
            self.tau**-0.5
        )
        xl_v = jnp.reshape(self.xl_v, [1, self.n_head, 1, self.d_k]) * (
            self.tau**-0.5
        )
        return xl_r, xl_u, xl_v

    def attn(self, present_q, present_k, present_v, present_doc_ids, state, vq_spec):
        bsz = present_q.shape[0]
        dims = chex.Dimensions(
            B=bsz,
            L=self.block_len,
            M=self.mem_len,
            W=self.mem_len + self.block_len,
            H=self.n_head,
            S=self.n_code,
            K=self.d_k,
            V=self.d_v,
            i=1,
        )
        chex.assert_trees_all_equal_dtypes(
            present_v,
            state["xlcache"]["v"],
            state["aggcache"]["upper_div_lower"],
            state["aggcache"]["lower"],
        )
        chex.assert_shape(present_q, dims["BHLK"])
        chex.assert_shape(present_k, dims["BHLK"])
        chex.assert_shape(present_v, dims["BHLV"])

        # quantize keys, compute metrics, commit loss, and codebook surrogate loss
        vq_output_dict = self.quantizer(present_k, vq_spec=vq_spec)
        present_z = vq_output_dict["shortcodes"]
        present_k_hat = vq_output_dict["quantized"]
        l_commit = vq_output_dict["l_commit"]
        l_codebook = vq_output_dict["l_codebook"]
        metrics = vq_output_dict["metrics"]
        chex.assert_shape(present_z, dims["BHL"])
        chex.assert_shape(present_k_hat, dims["BHLK"])
        chex.assert_trees_all_equal_dtypes(present_k, present_k_hat)

        # concatenate sliding window cache k/v onto current block
        xlcache = state["xlcache"]
        aggcache = state["aggcache"]
        chex.assert_shape(xlcache["z"], dims["BHM"])
        chex.assert_shape(xlcache["k_hat"], dims["BHMK"])
        chex.assert_shape(xlcache["v"], dims["BHMV"])
        recent_z = jnp.concatenate([xlcache["z"], present_z], axis=-1)
        recent_k_hat = jnp.concatenate([xlcache["k_hat"], present_k_hat], axis=-2)
        recent_v = jnp.concatenate([xlcache["v"], present_v], axis=-2)
        recent_doc_ids = jnp.concatenate([xlcache["doc_ids"], present_doc_ids], axis=-1)
        chex.assert_shape(recent_z, dims["BHW"])
        chex.assert_shape(recent_k_hat, dims["BHWK"])
        chex.assert_shape(recent_v, dims["BHWV"])

        # compute xl bias helpers
        xl_r, xl_u, xl_v = self.get_xl_helpers()

        # compute aggcache scores
        c = self.quantizer.get_codebook()
        cache_scores = jnp.einsum("bhlk,hsk->bhls", present_q + xl_u, c)
        cache_biases = VQAttention.get_agg_biases(aggcache["lower"])
        cache_biases = jnp.expand_dims(cache_biases, axis=-2)
        cache_scores += cache_biases

        # compute recent scores (present and xlcache)
        recent_scores_ac = jnp.einsum("bhlk,bhwk->bhlw", present_q + xl_u, recent_k_hat)
        recent_scores_bd = jnp.einsum("bhlk,hwk->bhlw", present_q + xl_v, xl_r)
        recent_scores_bd = VQAttention.rel_shift(recent_scores_bd)
        recent_scores_bd *= VQAttention.get_causal_mask(
            block_len=self.block_len,
            mem_len=self.mem_len,
            invalid_len=jax.nn.relu(self.mem_len - state["pos_offset"]),
            with_locality=True,  # zero out the dynamic biases in bubble
        )[None, None, ...].astype(jnp.int32)
        recent_scores = recent_scores_ac + recent_scores_bd
        keep_mask = VQAttention.get_causal_mask(
            block_len=self.block_len,
            mem_len=self.mem_len,
            invalid_len=jax.nn.relu(self.mem_len - state["pos_offset"]),
            with_locality=not self.agg_cache,  # attend to bubble when agg cache
        )[None, None, ...].astype(jnp.int32)
        recent_scores = recent_scores * keep_mask - MASK_INFTY_APPROX * (1 - keep_mask)

        # subtract max score for stability (ok due to softmax shift-invariance)
        cache_max_scores = jnp.max(cache_scores, axis=-1)  # BHL
        recent_max_scores = jnp.max(recent_scores, axis=-1)  # BHL
        max_scores = sg(jnp.maximum(cache_max_scores, recent_max_scores))  # BHL
        chex.assert_shape(max_scores, dims["BHL"])
        cache_scores -= max_scores[..., None]
        recent_scores -= max_scores[..., None]
        cache_a = jnp.exp(cache_scores).astype(self.dtype)
        recent_a = jnp.exp(recent_scores).astype(self.dtype)
        chex.assert_shape(cache_a, dims["BHLS"])
        chex.assert_shape(recent_a, dims["BHLW"])

        # compute per-query normalizer d and divide unnormalized weights a by it first,
        # so the numerically unstable expression av is never materialized.
        d = jnp.sum(recent_a, axis=-1)
        if self.agg_cache:
            d += jnp.sum(cache_a, axis=-1)
        wv = jnp.einsum("bhlw,bhwv->bhlv", recent_a / d[..., None], recent_v)
        if self.agg_cache:
            wv += jnp.einsum(
                "bhls,bhsv->bhlv", cache_a / d[..., None], aggcache["upper_div_lower"]
            )

        wv = jnp.transpose(wv, (0, 2, 1, 3))
        wv = jnp.reshape(wv, [bsz, self.block_len, self.n_head * self.d_v])
        return dict(
            attn_out=wv,
            recent_z=recent_z,
            recent_k_hat=recent_k_hat,
            recent_v=recent_v,
            recent_doc_ids=recent_doc_ids,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
        )

    def update_state(self, recent_z, recent_k_hat, recent_v, recent_doc_ids, state):
        bsz, *_ = recent_z.shape
        dims = chex.Dimensions(
            B=bsz,
            L=self.block_len,
            M=self.mem_len,
            S=self.n_code,
            H=self.n_head,
            K=self.d_k,
            V=self.d_v,
        )
        aggcache = state["aggcache"]
        chex.assert_shape(aggcache["upper_div_lower"], dims["BHSV"])
        chex.assert_shape(aggcache["lower"], dims["BHS"])
        chex.assert_shape(recent_z[..., : -self.mem_len], dims["BHL"])
        chex.assert_shape(recent_v[..., : -self.mem_len, :], dims["BHLV"])
        chex.assert_shape(recent_k_hat[..., : -self.mem_len, :], dims["BHLK"])
        chex.assert_shape(recent_z[..., -self.mem_len :], dims["BHM"])
        chex.assert_shape(recent_v[..., -self.mem_len :, :], dims["BHMV"])
        chex.assert_shape(recent_k_hat[..., -self.mem_len :, :], dims["BHMK"])
        # compute kronecker deltas; invalid z's from xlcache init encode to zero vecs
        delta = jax.nn.one_hot(
            recent_z[..., : -self.mem_len],
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
            jnp.einsum("bhls,bhlv->bhsv", f2, recent_v[..., : -self.mem_len, :]),
        )
        new_state = dict(
            pos_offset=new_pos_offset,
            xlcache=dict(
                z=recent_z[..., -self.mem_len :],
                k_hat=recent_k_hat[..., -self.mem_len :, :],
                v=recent_v[..., -self.mem_len :, :],
                doc_ids=recent_doc_ids[..., -self.mem_len :],
            ),
            aggcache=dict(
                lower=new_lower,
                upper_div_lower=new_upper_div_lower,
                latest_doc_id=recent_doc_ids[..., -self.mem_len - 1],
            ),
        )
        if not self.grad_thru_cache:
            new_state = jax.tree_util.tree_map(sg, new_state)
        return new_state

    def __call__(self, state, input_dict):
        doc_ids = input_dict.pop("doc_ids")
        vq_spec = input_dict.pop("vq_spec")
        x = input_dict.pop("input_features")
        x_tilde = self.input_ln(x)
        q = self.get_q(x_tilde=x_tilde)
        k, v, g = self.get_kvg(x_tilde=x_tilde)
        attn_output_dict = self.attn(q, k, v, doc_ids, state, vq_spec)
        wv = attn_output_dict.get("attn_out")
        o = wv * g
        res = self.res_proj(o)
        res = self.dropres(res)
        chex.assert_trees_all_equal_dtypes(x, res)
        new_state = self.update_state(
            recent_z=attn_output_dict.get("recent_z"),
            recent_k_hat=attn_output_dict.get("recent_k_hat"),
            recent_v=attn_output_dict.get("recent_v"),
            recent_doc_ids=attn_output_dict.get("recent_doc_ids"),
            state=state,
        )
        output_dict = dict(
            res=res,
            metrics=attn_output_dict.get("metrics"),
            l_commit=attn_output_dict.get("l_commit"),
            l_codebook=attn_output_dict.get("l_codebook"),
        )
        return new_state, output_dict
