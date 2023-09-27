"""
Helper class for VQ Attention.

Contains mostly static methods (for ease of unit testing).
"""
import dataclasses

import chex
import flax.linen as nn
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
import jax.scipy as jsp
from flax import struct

from transformer_vq.nn.grad import sg
from transformer_vq.nn.grad import st
from transformer_vq.nn.norm import LayerNorm
from transformer_vq.nn.pe import get_sinusoid_embs
from transformer_vq.nn.types import TransformerConfig


@struct.dataclass
class VQSpec:
    n_device: jax.Array
    n_block_per_update: jax.Array
    loss_mask: jax.Array

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in dataclasses.fields(VQSpec)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}
        return cls(**filtered)


def get_shortcodes(vecs, codebook):
    dims = chex.Dimensions(
        B=vecs.shape[0],
        H=vecs.shape[1],
        L=vecs.shape[2],
        S=codebook.shape[1],
        K=codebook.shape[2],
        i=1,
    )
    chex.assert_shape(vecs, dims["BHLK"])
    chex.assert_shape(codebook, dims["HSK"])
    diffs2 = (
        jnp.expand_dims(jnp.sum(jnp.square(vecs), axis=-1), -1)
        - 2.0 * jnp.einsum("bhlk,hsk->bhls", vecs, codebook)
        + jnp.expand_dims(jnp.sum(jnp.square(codebook), axis=-1), (0, 2))
    )  # B, H, L, S
    z = jnp.argmin(diffs2, axis=-1)
    chex.assert_shape(z, dims["BHL"])
    errs2 = jnp.min(diffs2, axis=-1)
    errs2 = jax.nn.relu(errs2)  # this is a no-op if using infinite precision
    chex.assert_shape(errs2, dims["BHL"])
    return z.astype(jnp.int32), errs2


def get_codewords(shortcodes, codebook):
    dims = chex.Dimensions(
        B=shortcodes.shape[0],
        H=shortcodes.shape[1],
        L=shortcodes.shape[2],
        S=codebook.shape[1],
        d=codebook.shape[2],
        i=1,
    )
    shortcodes = shortcodes[..., None]
    codebook = codebook[None, ...]
    chex.assert_shape(shortcodes, dims["BHLi"])
    chex.assert_shape(codebook, dims["iHSd"])
    cz = jnp.take_along_axis(codebook, indices=shortcodes, axis=2)
    return cz


class LearnableVQ(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        cs_args = [self.w_init, [self.n_head, self.n_code, self.d_k], self.param_dtype]
        cc_args = [init.ones, [self.n_head, self.n_code], self.param_dtype]
        self.c_sum = self.param("c_sum", *cs_args)
        self.c_count = self.param("c_count", *cc_args)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def _get_codebook(c_sum, c_count):
        c = c_sum / jnp.clip(c_count[..., None], a_min=0.01)
        return sg(c)

    def get_codebook(self):
        return LearnableVQ._get_codebook(self.c_sum, self.c_count)

    @staticmethod
    def get_codebook_ema_targets(vecs, shortcodes, c_sum, c_count, c_gamma, vq_spec):
        n_code = c_sum.shape[1]
        dims = chex.Dimensions(
            B=vecs.shape[0],
            L=vecs.shape[2],
            H=vecs.shape[1],
            d=vecs.shape[-1],
            S=c_sum.shape[1],
            i=1,
        )
        chex.assert_shape(vecs, dims["BHLd"])
        chex.assert_shape(shortcodes, dims["BHL"])
        chex.assert_shape(c_sum, dims["HSd"])
        chex.assert_shape(c_count, dims["HS"])
        chex.assert_shape(vq_spec.loss_mask, dims["BL"])
        g = c_gamma
        d = vq_spec.n_device
        p = vq_spec.n_block_per_update
        chex.assert_shape(d, dims["i"])
        chex.assert_shape(p, dims["i"])
        # we can compute running stats grouped by shortcode using r, see below.
        # since we also want to exclude vecs made using right-pad tokens, we
        # use loss_mask as heuristic to drop all such vecs from running stats.
        r = jax.nn.one_hot(shortcodes, num_classes=n_code, dtype=vecs.dtype)
        r *= jnp.expand_dims(jnp.expand_dims(vq_spec.loss_mask, 1), -1)
        c_sum_hat = d * p * jnp.einsum("bhts,bhtd->hsd", r, vecs)
        c_count_hat = d * p * jnp.sum(r, axis=(0, 2))
        c_sum_tgt = (1 - g) * c_sum_hat + g * c_sum
        c_count_tgt = (1 - g) * c_count_hat + g * c_count
        chex.assert_shape(c_sum_tgt, dims["HSd"])
        chex.assert_shape(c_count_tgt, dims["HS"])
        return c_sum_tgt, c_count_tgt

    @staticmethod
    def get_codebook_loss(
        vecs,
        shortcodes,
        c_sum,
        c_count,
        c_gamma,
        vq_spec,
    ):
        # the returned l_codebook gives correct updates
        # if gradients are averaged over devices and blocks
        # and codebook optimizer is sgd with lr = 1.0.
        batch_size = vecs.shape[0]
        n_head = vecs.shape[1]
        block_len = vecs.shape[2]
        d_k = vecs.shape[3]
        n_code = c_count.shape[1]
        dims = chex.Dimensions(B=batch_size, H=n_head, L=block_len, d=d_k, S=n_code)
        c_sum_tgt, c_count_tgt = LearnableVQ.get_codebook_ema_targets(
            vecs=vecs,
            shortcodes=shortcodes,
            c_sum=c_sum,
            c_count=c_count,
            c_gamma=c_gamma,
            vq_spec=vq_spec,
        )
        chex.assert_shape(c_sum_tgt, dims["HSd"])
        chex.assert_shape(c_count_tgt, dims["HS"])
        # fmt: on
        l_codebook_sum = jnp.sum(sg(c_sum - c_sum_tgt) * st(c_sum))
        l_codebook_count = jnp.sum(sg(c_count - c_count_tgt) * st(c_count))
        l_codebook = l_codebook_count + l_codebook_sum
        return l_codebook

    @staticmethod
    def get_quantization_metrics(vecs, vecs_hat, errs2, c_sum, c_count, dtype):
        # we'll call stop gradients in the return statement, so no need to call it now
        n_head, n_code = c_count.shape[0], c_count.shape[1]
        eps, errmin, errmax, maskval = 1e-2, 0e1, 1e1, 1e30
        c_count = jnp.clip(c_count, a_min=eps)
        c = c_sum / c_count[..., None]  # HSd
        c_norms = jnp.clip(jnp.linalg.norm(c, axis=-1), a_min=eps)  # HS
        c_normed = c / c_norms[..., None]  # HSd
        c_sims = jnp.einsum("hsd,hzd->hsz", c_normed, c_normed)  # HSS
        c_dists = jnp.linalg.norm(
            jnp.expand_dims(c, 2) - jnp.expand_dims(c, 1), axis=-1
        )  # HSS
        vec_norms = jnp.clip(jnp.linalg.norm(vecs, axis=-1), a_min=eps)  # BHL
        vec_hat_norms = jnp.clip(jnp.linalg.norm(vecs_hat, axis=-1), a_min=eps)  # BHL
        errs = jnp.sqrt(errs2)  # BHL
        relative_errs = jnp.clip(errs / vec_norms, errmin, errmax)  # BHL
        probs = c_count / jnp.sum(c_count, axis=-1)[..., None]  # HS
        c_thresh_oob = jnp.logical_or(c_count < 1.0, 1_000_000 < c_count)
        c_thresh_oob = c_thresh_oob.astype(jnp.float32)

        # elements will have shape [], [H] or [B, H], then we will tree map
        # to avg over heads/device batch items
        ones = jnp.ones([1, n_code, n_code], dtype=jnp.float32)
        up = jnp.triu(ones)  # upper triangular ones mask
        low = jnp.tril(ones, k=-1)  # strict lower triangular ones mask
        metrics = dict(
            c_sim_min=jnp.min(low * c_sims + maskval * up, axis=(1, 2)),  # [H]
            c_sim_mean=jnp.sum(low * c_sims, axis=(1, 2)) / jnp.sum(low, axis=(1, 2)),
            c_sim_max=jnp.max(low * c_sims - maskval * up, axis=(1, 2)),  # [H]
            c_dist_min=jnp.min(low * c_dists + maskval * up, axis=(1, 2)),  # [H]
            c_dist_mean=jnp.sum(low * c_dists, axis=(1, 2)) / jnp.sum(low, axis=(1, 2)),
            c_dist_max=jnp.max(low * c_dists - maskval * up, axis=(1, 2)),  # [H]
            c_norm_min=jnp.min(c_norms, axis=1),  # [H]
            c_norm_mean=jnp.mean(c_norms, axis=1),  # [H]
            c_norm_max=jnp.max(c_norms, axis=1),  # [H]
            c_usage_min=jnp.min(c_count, axis=1),  # [H]
            c_usage_mean=jnp.mean(c_count, axis=1),  # [H]
            c_usage_max=jnp.max(c_count, axis=1),  # [H]
            c_thresh_oob=jnp.sum(c_thresh_oob, axis=1),  # [H]
            c_entropy=jnp.sum(jsp.special.entr(probs), axis=-1),  # [H]
            vec_norm_mean=jnp.mean(vec_norms, axis=2),  # [B, H]
            vec_hat_norm_mean=jnp.mean(vec_hat_norms, axis=2),  # [B, H]
            relative_err_min=jnp.min(relative_errs, axis=2),  # [B, H]
            relative_err_mean=jnp.mean(relative_errs, axis=2),  # [B, H]
            relative_err_max=jnp.max(relative_errs, axis=2),  # [B, H]
        )
        return jax.tree_util.tree_map(lambda x: jnp.mean(sg(x)).astype(dtype), metrics)

    def __call__(self, vecs, vq_spec):
        orig_dtype = vecs.dtype
        vecs_hp = vecs.astype(self.param_dtype)
        c = LearnableVQ._get_codebook(self.c_sum, self.c_count)
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        errs2 = errs2.astype(self.dtype)
        cz = get_codewords(shortcodes=z, codebook=c)
        cz = cz.astype(orig_dtype)
        vecs_hat = sg(cz) + st(vecs)
        if self.is_train:
            loss_mask = vq_spec.loss_mask
            l_commit = jnp.mean(jnp.sum(jnp.expand_dims(loss_mask, 1) * errs2, axis=1))
            l_codebook = LearnableVQ.get_codebook_loss(
                vecs=vecs_hp,
                shortcodes=z,
                c_sum=self.c_sum,
                c_count=self.c_count,
                c_gamma=self.c_gamma,
                vq_spec=vq_spec,
            ).astype(self.dtype)
        else:
            l_commit = jnp.zeros(dtype=self.dtype, shape=[])
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
        if self.is_train:
            metrics = LearnableVQ.get_quantization_metrics(
                vecs=sg(vecs),
                vecs_hat=sg(vecs_hat),
                errs2=sg(errs2),
                c_sum=sg(self.c_sum),
                c_count=sg(self.c_count),
                dtype=self.dtype,
            )
        else:
            metrics = dict()
        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
            errs2=errs2,
        )


class SimpleVQ(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        self.tau = self.d_k**0.5
        self.norm = LayerNorm(
            input_dim=self.d_k,
            param_dtype=self.param_dtype,
            center=False,
            norm=True,
            gain=False,
            bias=False,
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def get_codebook(self):
        c = get_sinusoid_embs(
            length=self.n_code, width=self.d_k, start=0, lam=self.pe_lam, flip=False
        )
        return (self.tau**-0.5) * sg(self.norm(c))[None, ...]

    def __call__(self, vecs, vq_spec):
        orig_dtype = vecs.dtype
        vecs_hp = vecs.astype(self.param_dtype)
        c = self.get_codebook()
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        errs2 = errs2.astype(self.dtype)
        cz = get_codewords(shortcodes=z, codebook=c)
        cz = cz.astype(orig_dtype)
        vecs_hat = sg(cz) + st(vecs)
        if self.is_train:
            loss_mask = vq_spec.loss_mask
            l_commit = jnp.mean(jnp.sum(jnp.expand_dims(loss_mask, 1) * errs2, axis=1))
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
        else:
            l_commit = jnp.zeros(dtype=self.dtype, shape=[])
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
        metrics = dict()
        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
            errs2=errs2,
        )
