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

from transformer_vq.nn.types import TransformerConfig


def sg(x):
    return jax.lax.stop_gradient(x)


def st(x):
    return x - jax.lax.stop_gradient(x)


def get_shortcodes(vecs, codebook):
    dims = chex.Dimensions(
        B=vecs.shape[0],
        H=vecs.shape[1],
        R=vecs.shape[2],
        C=vecs.shape[3],
        S=codebook.shape[1],
        K=codebook.shape[2],
        i=1,
    )
    chex.assert_shape(vecs, dims["BHRCK"])
    chex.assert_shape(codebook, dims["HSK"])
    diffs2 = (
        jnp.einsum("bhrck->bhrc", jnp.square(vecs))[..., None]
        - 2.0 * jnp.einsum("bhrck,hsk->bhrcs", vecs, codebook)
        + jnp.expand_dims(jnp.einsum("hsk->hs", jnp.square(codebook)), (0, 2, 3))
    )
    z = jnp.argmin(diffs2, axis=-1)
    chex.assert_shape(z, dims["BHRC"])
    errs2 = jnp.min(diffs2, axis=-1)
    errs2 = jax.nn.relu(errs2)  # this is a no-op if using infinite precision
    chex.assert_shape(errs2, dims["BHRC"])
    return z.astype(jnp.int32), errs2


def get_codewords(shortcodes, codebook):
    dims = chex.Dimensions(
        B=shortcodes.shape[0],
        H=shortcodes.shape[1],
        R=shortcodes.shape[2],
        C=shortcodes.shape[3],
        S=codebook.shape[1],
        K=codebook.shape[2],
        i=1,
    )
    chex.assert_shape(shortcodes, dims["BHRC"])
    chex.assert_shape(codebook, dims["HSK"])
    shortcodes = jnp.expand_dims(shortcodes, -1)
    codebook = jnp.expand_dims(codebook, (0, 2))
    chex.assert_shape(shortcodes, dims["BHRCi"])
    chex.assert_shape(codebook, dims["iHiSK"])
    cz = jnp.take_along_axis(codebook, shortcodes, axis=-2)
    return cz


class VectorQuantizer(nn.Module):
    config: TransformerConfig
    n_head: int

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
    def _get_codebook(c_sum, c_count, dtype):
        c = c_sum / jnp.clip(c_count[..., None], a_min=0.01)
        return sg(c).astype(dtype)

    def get_codebook(self):
        return VectorQuantizer._get_codebook(self.c_sum, self.c_count, self.dtype)

    @staticmethod
    def get_codebook_ema_targets(
        vecs, shortcodes, c_sum, c_count, c_gamma, n_device, n_block_per_update
    ):
        n_code = c_sum.shape[1]
        dims = chex.Dimensions(
            B=vecs.shape[0],
            H=vecs.shape[1],
            R=vecs.shape[2],
            C=vecs.shape[3],
            S=c_sum.shape[1],
            K=c_sum.shape[2],
            i=1,
        )
        chex.assert_shape(vecs, dims["BHRCK"])
        chex.assert_shape(shortcodes, dims["BHRC"])
        chex.assert_shape(c_sum, dims["HSK"])
        chex.assert_shape(c_count, dims["HS"])
        g = c_gamma
        d = n_device
        p = n_block_per_update
        chex.assert_shape(d, dims["i"])
        chex.assert_shape(p, dims["i"])
        # we can compute running stats grouped by shortcode using r, see below.
        # since we also want to exclude vecs made using right-pad tokens, we
        # use loss_mask as heuristic to drop all such vecs from running stats.
        r = jax.nn.one_hot(shortcodes, num_classes=n_code, dtype=vecs.dtype)
        c_sum_hat = d * p * jnp.einsum("bhrcs,bhrck->hsk", r, vecs)
        c_count_hat = d * p * jnp.einsum("bhrcs->hs", r)
        c_sum_tgt = (1 - g) * c_sum_hat + g * c_sum
        c_count_tgt = (1 - g) * c_count_hat + g * c_count
        chex.assert_shape(c_sum_tgt, dims["HSK"])
        chex.assert_shape(c_count_tgt, dims["HS"])
        return c_sum_tgt, c_count_tgt

    @staticmethod
    def get_codebook_loss(
        vecs,
        shortcodes,
        c_sum,
        c_count,
        c_gamma,
        n_device,
        n_block_per_update,
    ):
        # the returned l_codebook gives correct updates
        # if gradients are averaged over devices and blocks
        # and codebook optimizer is sgd with lr = 1.0.
        c_sum_tgt, c_count_tgt = VectorQuantizer.get_codebook_ema_targets(
            vecs=vecs,
            shortcodes=shortcodes,
            c_sum=c_sum,
            c_count=c_count,
            c_gamma=c_gamma,
            n_device=n_device,
            n_block_per_update=n_block_per_update,
        )
        l_codebook_sum = jnp.sum(sg(c_sum - c_sum_tgt) * st(c_sum))
        l_codebook_count = jnp.sum(sg(c_count - c_count_tgt) * st(c_count))
        l_codebook = l_codebook_count + l_codebook_sum
        return l_codebook

    def __call__(self, vecs):
        orig_dtype = vecs.dtype
        vecs_hp = vecs.astype(jnp.float32)
        c = VectorQuantizer._get_codebook(self.c_sum, self.c_count, jnp.float32)
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        errs2 = errs2.astype(self.dtype)
        cz = get_codewords(shortcodes=z, codebook=c)
        cz = cz.astype(orig_dtype)
        vecs_hat = sg(cz) + st(vecs)  # exact by sterbenz's lemma
        if self.is_train:
            l_commit = jnp.mean(jnp.sum(errs2, axis=1)).astype(self.dtype)  # sum heads
            l_codebook = VectorQuantizer.get_codebook_loss(
                vecs=vecs_hp,
                shortcodes=z,
                c_sum=self.c_sum,
                c_count=self.c_count,
                c_gamma=self.c_gamma,
                n_device=jnp.array([self.n_device]),
                n_block_per_update=jnp.array([self.sequence_len // self.block_len]),
            ).astype(self.dtype)
        else:
            l_commit = jnp.zeros(dtype=self.dtype, shape=[])
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            errs2=errs2,
        )
