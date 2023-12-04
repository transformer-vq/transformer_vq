import dataclasses

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from transformer_vq.nn.norm import RMSNorm
from transformer_vq.nn.pe import XLBiasProducer
from transformer_vq.nn.qkv import QKVGProducer
from transformer_vq.nn.sharding import sharding_constraint
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import EMAVectorQuantizer

INFTY_APPROX = 1e30  # large value approximating infinity


class VQAttention(nn.Module):
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

    def setup(self):
        self.apply_config()
        assert self.sequence_len // self.block_len > 2
        self.tau = self.d_k**0.5
        self.input_ln = RMSNorm()
        self.qkvg_producer = QKVGProducer(self.config, self.global_mesh)
        self.quantizer = EMAVectorQuantizer(self.config, self.global_mesh)
        self.xl_bias_producer = XLBiasProducer(self.config, self.global_mesh)
        proj_kwargs = dict(
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res_proj = nn.Dense(
            self.d_model,
            **proj_kwargs,
            kernel_init=nn.with_partitioning(
                self.w_init, names=("model", None), mesh=self.global_mesh
            ),
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

    def get_cache_vars_serial(self, z: jax.Array, v: jax.Array, dims: chex.Dimensions):
        # this function computes our per-block cache variables using a cross-block
        # cumulative reductions, similar to hua et al., 2022 but more stable
        delta = jax.nn.one_hot(z, num_classes=self.n_code, dtype=self.dtype, axis=-1)
        delta = sharding_constraint(delta, self.global_mesh, ("data", None, None, None))
        chex.assert_shape(delta, dims["BRCS"])
        n_block = delta.shape[1]
        delta1_by_block = jnp.einsum("brcs->brs", delta)
        deltav_by_block = jnp.einsum("brcs,brcv->brsv", delta, v)
        delta1_by_block = sharding_constraint(
            delta1_by_block, self.global_mesh, ("data", None, None)
        )
        deltav_by_block = sharding_constraint(
            deltav_by_block, self.global_mesh, ("data", None, None, "model")
        )
        deltav_by_block_normalized = jnp.divide(
            deltav_by_block,
            jnp.clip(delta1_by_block[..., None], a_min=1.0),
        )
        deltav_by_block_normalized = sharding_constraint(
            deltav_by_block_normalized, self.global_mesh, ("data", None, None, "model")
        )

        def scan_loop_body(carry, in_dict):
            old_lower = carry["lower"]
            new_lower = old_lower + in_dict["delta1_by_block"]
            f1 = jnp.divide(old_lower, jnp.clip(new_lower, a_min=1.0))
            f2 = jnp.divide(in_dict["delta1_by_block"], jnp.clip(new_lower, a_min=1.0))
            new_upper_div_lower = jnp.add(
                f1[..., None] * carry["upper_div_lower"],
                f2[..., None] * in_dict["deltav_by_block_normalized"],
            )
            new_lower = sharding_constraint(new_lower, self.global_mesh, ("data", None))
            new_upper_div_lower = sharding_constraint(
                new_upper_div_lower, self.global_mesh, ("data", None, "model")
            )
            carry_new = dict(
                upper_div_lower=new_upper_div_lower,
                lower=new_lower,
            )
            return carry_new, carry_new

        _, cache_vars = jax.lax.scan(
            f=scan_loop_body,
            init=dict(
                upper_div_lower=sharding_constraint(
                    jnp.zeros(dtype=self.dtype, shape=dims["BSV"]),
                    self.global_mesh,
                    ("data", None, "model"),
                ),
                lower=sharding_constraint(
                    jnp.zeros(dtype=self.dtype, shape=dims["BS"]),
                    self.global_mesh,
                    ("data", None),
                ),
            ),
            xs=dict(
                deltav_by_block_normalized=jnp.transpose(
                    deltav_by_block_normalized, (1, 0, 2, 3)
                ),
                delta1_by_block=jnp.transpose(delta1_by_block, (1, 0, 2)),
            ),
            length=n_block,
            unroll=1,
        )

        cache_var_upper_div_lower = jnp.pad(
            jnp.transpose(cache_vars["upper_div_lower"][:-2], (1, 0, 2, 3)),
            ((0, 0), (2, 0), (0, 0), (0, 0)),
        )
        cache_var_lower = jnp.pad(
            jnp.transpose(cache_vars["lower"][:-2], (1, 0, 2)),
            ((0, 0), (2, 0), (0, 0)),
        )
        return cache_var_upper_div_lower, cache_var_lower

    def compute_cache_vars(self, z, v, dims):
        if self.reduction_type == "serial":
            return self.get_cache_vars_serial(z, v, dims)
        raise NotImplementedError(f"Unknown reduction_type: {self.reduction_type}")

    def __call__(self, x):
        dims = chex.Dimensions(
            B=x.shape[0],
            T=self.sequence_len,
            R=self.sequence_len // self.block_len,
            C=self.block_len,
            W=2 * self.block_len,
            D=self.d_model,
            S=self.n_code,
            K=self.d_k,
            V=QKVGProducer.get_d_v(self.config),
        )
        x = sharding_constraint(x, self.global_mesh, ("data", None, None, None))
        x_tilde = self.input_ln(x)
        x_tilde = sharding_constraint(
            x_tilde, self.global_mesh, ("data", None, None, None)
        )
        q = self.qkvg_producer.get_queries(x_tilde=x_tilde)
        k, v = self.qkvg_producer.get_keys_and_values(x_tilde=x_tilde)
        q = sharding_constraint(q, self.global_mesh, ("data", None, None, None))
        k = sharding_constraint(k, self.global_mesh, ("data", None, None, None))
        v = sharding_constraint(v, self.global_mesh, ("data", None, None, "model"))
        chex.assert_shape(q, dims["BRCK"])
        chex.assert_shape(k, dims["BRCK"])
        chex.assert_shape(v, dims["BRCV"])
        chex.assert_trees_all_equal_dtypes(q, k, v)

        vq_out = self.quantizer(k)
        z = vq_out["shortcodes"]
        k_hat = vq_out["quantized"]
        l_commit = vq_out["l_commit"]
        l_codebook = vq_out["l_codebook"]
        z = sharding_constraint(z, self.global_mesh, ("data", None, None))
        k_hat = sharding_constraint(k_hat, self.global_mesh, ("data", None, None, None))
        l_commit = sharding_constraint(l_commit, self.global_mesh, ())
        l_codebook = sharding_constraint(l_codebook, self.global_mesh, ())
        chex.assert_shape(z, dims["BRC"])
        chex.assert_shape(k_hat, dims["BRCK"])
        chex.assert_shape(l_commit, [])
        chex.assert_shape(l_codebook, [])
        chex.assert_trees_all_equal_dtypes(k, k_hat)

        xl_r, xl_u, xl_v = self.xl_bias_producer(k_len=2 * self.block_len)
        xl_r = sharding_constraint(xl_r, self.global_mesh, (None, None))
        xl_u = sharding_constraint(xl_u, self.global_mesh, (None,))
        xl_v = sharding_constraint(xl_v, self.global_mesh, (None,))
        chex.assert_shape(xl_r, dims["WK"])
        chex.assert_shape(xl_u, dims["K"])
        chex.assert_shape(xl_v, dims["K"])
        chex.assert_trees_all_equal_dtypes(xl_r, xl_u, xl_v)
        q_plus_xlu = q + xl_u[None, None, None, ...]
        q_plus_xlv = q + xl_v[None, None, None, ...]
        q_plus_xlu = sharding_constraint(
            q_plus_xlu, self.global_mesh, ("data", None, None, None)
        )
        q_plus_xlv = sharding_constraint(
            q_plus_xlv, self.global_mesh, ("data", None, None, None)
        )
        chex.assert_shape(q_plus_xlu, dims["BRCK"])
        chex.assert_shape(q_plus_xlv, dims["BRCK"])
        chex.assert_trees_all_equal_dtypes(q, q_plus_xlu, q_plus_xlv)

        k_hat_prev = jnp.pad(k_hat[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0)))
        v_prev = jnp.pad(v[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0)))
        codebook = self.quantizer.get_codebook()
        k_hat_prev = sharding_constraint(
            k_hat_prev, self.global_mesh, ("data", None, None, None)
        )
        v_prev = sharding_constraint(
            v_prev, self.global_mesh, ("data", None, None, "model")
        )
        codebook = sharding_constraint(codebook, self.global_mesh, (None, None))
        chex.assert_shape(k_hat_prev, dims["BRCK"])
        chex.assert_shape(v_prev, dims["BRCV"])
        chex.assert_shape(codebook, dims["SK"])
        chex.assert_trees_all_equal_dtypes(k_hat_prev, v_prev, codebook)

        scores_ac_present = jnp.einsum("brik,brjk->brij", q_plus_xlu, k_hat)
        scores_ac_prev = jnp.einsum("brik,brjk->brij", q_plus_xlu, k_hat_prev)
        scores_cache = jnp.einsum("brck,sk->brcs", q_plus_xlu, codebook)
        scores_ac_present = sharding_constraint(
            scores_ac_present, self.global_mesh, ("data", None, None, None)
        )
        scores_ac_prev = sharding_constraint(
            scores_ac_prev, self.global_mesh, ("data", None, None, None)
        )
        scores_cache = sharding_constraint(
            scores_cache, self.global_mesh, ("data", None, None, None)
        )
        chex.assert_shape(scores_ac_present, dims["BRCC"])
        chex.assert_shape(scores_ac_prev, dims["BRCC"])
        chex.assert_shape(scores_cache, dims["BRCS"])
        chex.assert_trees_all_equal_dtypes(
            scores_ac_present, scores_ac_prev, scores_cache
        )

        scores_bd_recent = jnp.einsum("brck,wk->brcw", q_plus_xlv, xl_r)
        scores_bd_recent = sharding_constraint(
            scores_bd_recent, self.global_mesh, ("data", None, None, None)
        )
        scores_bd_recent = XLBiasProducer.rel_shift(
            scores_bd_recent, head_type=self.head_type, global_mesh=self.global_mesh
        )
        scores_bd_recent = sharding_constraint(
            scores_bd_recent, self.global_mesh, ("data", None, None, None)
        )
        scores_bd_recent *= XLBiasProducer.get_causal_keep_mask(
            q_len=self.block_len,
            k_len=2 * self.block_len,
            with_locality=True,  # zero out the biases left of the sliding window
            global_mesh=self.global_mesh,
        )[None, None, ...]
        scores_bd_recent = sharding_constraint(
            scores_bd_recent, self.global_mesh, ("data", None, None, None)
        )
        scores_bd_prev, scores_bd_present = jnp.split(scores_bd_recent, 2, axis=-1)
        scores_bd_prev = sharding_constraint(
            scores_bd_prev, self.global_mesh, ("data", None, None, None)
        )
        scores_bd_present = sharding_constraint(
            scores_bd_present, self.global_mesh, ("data", None, None, None)
        )
        chex.assert_shape(scores_bd_present, dims["BRCC"])
        chex.assert_shape(scores_bd_prev, dims["BRCC"])
        chex.assert_trees_all_equal_dtypes(scores_bd_present, scores_bd_prev)

        scores_present = scores_ac_present + scores_bd_present
        scores_prev = scores_ac_prev + scores_bd_prev
        scores_present = sharding_constraint(
            scores_present, self.global_mesh, ("data", None, None, None)
        )
        scores_prev = sharding_constraint(
            scores_prev, self.global_mesh, ("data", None, None, None)
        )
        chex.assert_shape(scores_present, dims["BRCC"])
        chex.assert_shape(scores_prev, dims["BRCC"])
        chex.assert_trees_all_equal_dtypes(scores_present, scores_prev)

        # mask out non-causal connections within each block
        mask_present = XLBiasProducer.get_causal_keep_mask(
            q_len=self.block_len,
            k_len=self.block_len,
            with_locality=False,
            global_mesh=self.global_mesh,
        )[None, None, ...]
        scores_present -= jnp.array(INFTY_APPROX, dtype=self.dtype) * jnp.logical_not(
            mask_present
        )
        scores_present = sharding_constraint(
            scores_present, self.global_mesh, ("data", None, None, None)
        )
        # mask out attn weight contrib from zero padding in k_prev
        scores_prev = jnp.pad(
            scores_prev[:, 1:],
            ((0, 0), (1, 0), (0, 0), (0, 0)),
            constant_values=-INFTY_APPROX,
        )
        scores_prev = sharding_constraint(
            scores_prev, self.global_mesh, ("data", None, None, None)
        )
        chex.assert_shape(scores_present, dims["BRCC"])
        chex.assert_trees_all_equal_dtypes(scores_present, scores_prev, scores_cache)

        cache_var_upper_div_lower, cache_var_lower = self.compute_cache_vars(z, v, dims)
        cache_var_upper_div_lower = sharding_constraint(
            cache_var_upper_div_lower, self.global_mesh, ("data", None, None, "model")
        )
        cache_var_lower = sharding_constraint(
            cache_var_lower, self.global_mesh, ("data", None, None)
        )
        count_biases = jnp.where(
            jnp.equal(cache_var_lower, jnp.zeros_like(cache_var_lower)),
            -INFTY_APPROX,
            jnp.log(jnp.maximum(cache_var_lower, jnp.ones_like(cache_var_lower))),
        )
        chex.assert_shape(count_biases, dims["BRS"])
        scores_cache += jnp.expand_dims(count_biases, -2)
        scores_cache = sharding_constraint(
            scores_cache, self.global_mesh, ("data", None, None, None)
        )
        chex.assert_shape(scores_cache, dims["BRCS"])

        # subtract max score for stability (ok due to softmax shift-invariance)
        scores_present_max = jnp.max(scores_present, axis=-1)
        scores_prev_max = jnp.max(scores_prev, axis=-1)
        scores_cache_max = jnp.max(scores_cache, axis=-1)
        scores_present_max = sharding_constraint(
            scores_present_max, self.global_mesh, ("data", None, None)
        )
        scores_prev_max = sharding_constraint(
            scores_prev_max, self.global_mesh, ("data", None, None)
        )
        scores_cache_max = sharding_constraint(
            scores_cache_max, self.global_mesh, ("data", None, None)
        )
        chex.assert_shape(scores_present_max, dims["BRC"])
        chex.assert_shape(scores_prev_max, dims["BRC"])
        chex.assert_shape(scores_cache_max, dims["BRC"])
        max_scores = jnp.maximum(
            jnp.maximum(scores_present_max, scores_prev_max), scores_cache_max
        )
        max_scores = jax.lax.stop_gradient(max_scores)
        scores_present -= max_scores[..., None]
        scores_prev -= max_scores[..., None]
        scores_cache -= max_scores[..., None]
        scores_present = sharding_constraint(
            scores_present, self.global_mesh, ("data", None, None, None)
        )
        scores_prev = sharding_constraint(
            scores_prev, self.global_mesh, ("data", None, None, None)
        )
        scores_cache = sharding_constraint(
            scores_cache, self.global_mesh, ("data", None, None, None)
        )

        # exponentiate the attention scores
        a_present = jnp.exp(scores_present)
        a_prev = jnp.exp(scores_prev)
        a_cache = jnp.exp(scores_cache)
        chex.assert_shape(a_present, dims["BRCC"])
        chex.assert_shape(a_prev, dims["BRCC"])
        chex.assert_shape(a_cache, dims["BRCS"])
        chex.assert_trees_all_equal_dtypes(a_present, a_prev, a_cache)

        # compute attn normalizer, attn weights, and attn outputs
        d = jnp.sum(a_present, axis=-1)
        d += jnp.sum(a_prev, axis=-1)
        d += jnp.sum(a_cache, axis=-1)
        d = sharding_constraint(d, self.global_mesh, ("data", None, None))
        w_present = a_present / d[..., None]
        w_prev = a_prev / d[..., None]
        w_cache = a_cache / d[..., None]
        w_present = sharding_constraint(
            w_present, self.global_mesh, ("data", None, None, None)
        )
        w_prev = sharding_constraint(
            w_prev, self.global_mesh, ("data", None, None, None)
        )
        w_cache = sharding_constraint(
            w_cache, self.global_mesh, ("data", None, None, None)
        )
        wv = jnp.einsum("brij,brjv->briv", w_present, v)
        wv += jnp.einsum("brij,brjv->briv", w_prev, v_prev)
        wv += jnp.einsum("bris,brsv->briv", w_cache, cache_var_upper_div_lower)
        wv = sharding_constraint(wv, self.global_mesh, ("data", None, None, "model"))
        chex.assert_shape(wv, dims["BRCV"])

        if self.head_type == "shga":
            wv *= self.qkvg_producer.get_gates(x_tilde=x_tilde)
        wv = sharding_constraint(wv, self.global_mesh, ("data", None, None, "model"))
        res = self.res_proj(wv)
        res = sharding_constraint(res, self.global_mesh, ("data", None, None, None))
        chex.assert_trees_all_equal_dtypes(res, wv, d, q, k, v, x, x_tilde)
        return dict(
            res=res,
            l_commit=l_commit,
            l_codebook=l_codebook,
        )


class FullAttention(nn.Module):
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

    def setup(self):
        self.apply_config()
        assert self.sequence_len // self.block_len > 2
        self.tau = self.d_k**0.5
        self.input_ln = RMSNorm()
        self.qkvg_producer = QKVGProducer(self.config, self.global_mesh)
        self.xl_bias_producer = XLBiasProducer(self.config, self.global_mesh)
        proj_kwargs = dict(
            use_bias=False,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.res_proj = nn.Dense(
            self.d_model,
            **proj_kwargs,
            kernel_init=nn.with_partitioning(
                self.w_init, names=("model", None), mesh=self.global_mesh
            ),
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
            K=self.d_k,
            V=QKVGProducer.get_d_v(self.config),
        )
        x = sharding_constraint(x, self.global_mesh, ("data", None, None))
        x_tilde = self.input_ln(x)
        x_tilde = sharding_constraint(x_tilde, self.global_mesh, ("data", None, None))
        q = self.qkvg_producer.get_queries(x_tilde=x_tilde)
        k, v = self.qkvg_producer.get_keys_and_values(x_tilde=x_tilde)
        q = sharding_constraint(q, self.global_mesh, ("data", None, None))
        k = sharding_constraint(k, self.global_mesh, ("data", None, None))
        v = sharding_constraint(v, self.global_mesh, ("data", None, "model"))
        chex.assert_shape(q, dims["BTK"])
        chex.assert_shape(k, dims["BTK"])
        chex.assert_shape(v, dims["BTV"])
        chex.assert_trees_all_equal_dtypes(q, k, v)

        xl_r, xl_u, xl_v = self.xl_bias_producer(k_len=self.sequence_len)
        xl_r = sharding_constraint(xl_r, self.global_mesh, (None, None))
        xl_u = sharding_constraint(xl_u, self.global_mesh, (None,))
        xl_v = sharding_constraint(xl_v, self.global_mesh, (None,))
        chex.assert_shape(xl_r, dims["TK"])
        chex.assert_shape(xl_u, dims["K"])
        chex.assert_shape(xl_v, dims["K"])
        chex.assert_trees_all_equal_dtypes(xl_r, xl_u, xl_v)
        q_plus_xlu = q + xl_u[None, None, ...]
        q_plus_xlv = q + xl_v[None, None, ...]
        q_plus_xlu = sharding_constraint(
            q_plus_xlu, self.global_mesh, ("data", None, None)
        )
        q_plus_xlv = sharding_constraint(
            q_plus_xlv, self.global_mesh, ("data", None, None)
        )
        chex.assert_shape(q_plus_xlu, dims["BTK"])
        chex.assert_shape(q_plus_xlv, dims["BTK"])
        chex.assert_trees_all_equal_dtypes(q, q_plus_xlu, q_plus_xlv)

        scores_ac = jnp.einsum("bik,bjk->bij", q_plus_xlu, k)
        scores_bd = jnp.einsum("bik,jk->bij", q_plus_xlv, xl_r)
        scores_ac = sharding_constraint(
            scores_ac, self.global_mesh, ("data", None, None)
        )
        scores_bd = sharding_constraint(
            scores_bd, self.global_mesh, ("data", None, None)
        )
        scores_bd = XLBiasProducer.rel_shift(
            scores_bd, head_type=self.head_type, global_mesh=self.global_mesh
        )
        scores_bd = sharding_constraint(
            scores_bd, self.global_mesh, ("data", None, None)
        )
        chex.assert_shape(scores_ac, dims["BTT"])
        chex.assert_shape(scores_bd, dims["BTT"])
        chex.assert_trees_all_equal_dtypes(scores_ac, scores_bd)

        scores = scores_ac + scores_bd
        scores = sharding_constraint(scores, self.global_mesh, ("data", None, None))
        mask = XLBiasProducer.get_causal_keep_mask(
            q_len=self.sequence_len,
            k_len=self.sequence_len,
            with_locality=False,
            global_mesh=self.global_mesh,
        )[None, ...]
        scores -= jnp.array(INFTY_APPROX, dtype=self.dtype) * jnp.logical_not(mask)
        scores = sharding_constraint(scores, self.global_mesh, ("data", None, None))

        w = jax.nn.softmax(scores, axis=-1)
        w = sharding_constraint(w, self.global_mesh, ("data", None, None))
        wv = jnp.einsum("bij,bjv->biv", w, v)
        wv = sharding_constraint(wv, self.global_mesh, ("data", None, "model"))
        chex.assert_shape(w, dims["BTT"])
        chex.assert_shape(wv, dims["BTV"])

        if self.head_type == "shga":
            wv *= self.qkvg_producer.get_gates(x_tilde=x_tilde)
        wv = sharding_constraint(wv, self.global_mesh, ("data", None, "model"))
        res = self.res_proj(wv)
        res = sharding_constraint(res, self.global_mesh, ("data", None, None))
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
    raise NotImplementedError(f"Unknown attention type {attn_type}")
