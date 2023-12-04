"""
Helper class for VQ Attention.

Contains mostly static methods (for ease of unit testing).
"""
import dataclasses

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from transformer_vq.nn.sharding import sharding_constraint
from transformer_vq.nn.types import TransformerConfig


def sg(x):
    return jax.lax.stop_gradient(x)


def st(x):
    return jnp.subtract(x, jax.lax.stop_gradient(x))


def get_shortcodes(vecs, codebook, global_mesh):
    dims = chex.Dimensions(
        B=vecs.shape[0],
        R=vecs.shape[1],
        C=vecs.shape[2],
        S=codebook.shape[0],
        K=codebook.shape[1],
        i=1,
    )
    vecs = sharding_constraint(vecs, global_mesh, ("data", None, None, None))
    codebook = sharding_constraint(codebook, global_mesh, (None, None))
    chex.assert_shape(vecs, dims["BRCK"])
    chex.assert_shape(codebook, dims["SK"])

    diffs2 = (
        jnp.einsum("brck->brc", jnp.square(vecs))[..., None]
        + -2.0 * jnp.einsum("brck,sk->brcs", vecs, codebook)
        + jnp.einsum("sk->s", jnp.square(codebook))[None, None, None, ...]
    )
    z = jnp.argmin(diffs2, axis=-1)
    z = sharding_constraint(z, global_mesh, ("data", None, None))
    chex.assert_shape(z, dims["BRC"])
    errs2 = jnp.min(diffs2, axis=-1)
    errs2 = sharding_constraint(errs2, global_mesh, ("data", None, None))
    errs2 = jax.nn.relu(errs2)  # this is a no-op if using infinite precision
    errs2 = sharding_constraint(errs2, global_mesh, ("data", None, None))
    chex.assert_shape(errs2, dims["BRC"])
    return z.astype(jnp.int32), errs2


def get_codewords(shortcodes, codebook, global_mesh):
    dims = chex.Dimensions(
        B=shortcodes.shape[0],
        R=shortcodes.shape[1],
        C=shortcodes.shape[2],
        S=codebook.shape[0],
        K=codebook.shape[1],
        i=1,
    )
    shortcodes = sharding_constraint(shortcodes, global_mesh, ("data", None, None))
    codebook = sharding_constraint(codebook, global_mesh, (None, None))
    chex.assert_shape(shortcodes, dims["BRC"])
    chex.assert_shape(codebook, dims["SK"])
    shortcodes = shortcodes[..., None]
    codebook = codebook[None, None, ...]
    shortcodes = sharding_constraint(
        shortcodes, global_mesh, ("data", None, None, None)
    )
    codebook = sharding_constraint(codebook, global_mesh, (None, None, None, None))
    chex.assert_shape(shortcodes, dims["BRCi"])
    chex.assert_shape(codebook, dims["iiSK"])
    cz = jnp.take_along_axis(codebook, shortcodes, axis=-2)
    cz = sharding_constraint(cz, global_mesh, ("data", None, None, None))
    return cz


class SimpleVectorQuantizer(nn.Module):
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

    def setup(self):
        self.apply_config()
        self.codebook = self.param(
            "codebook",
            self.w_init,
            [self.n_code, self.d_k],
            self.param_dtype,
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def _get_codebook(c, dtype):
        return sg(c).astype(dtype)

    def get_codebook(self):
        return SimpleVectorQuantizer._get_codebook(self.codebook, self.dtype)

    def __call__(self, vecs):
        assert self.head_type == "shga", "Need to edit things for tp if using mha/mqa"
        orig_dtype = vecs.dtype
        vecs_hp = vecs.astype(jnp.float32)
        c = self.codebook
        z, losses_commit = get_shortcodes(
            vecs=vecs_hp, codebook=sg(c), global_mesh=self.global_mesh
        )
        _, losses_codebook = get_shortcodes(
            vecs=sg(vecs_hp), codebook=c, global_mesh=self.global_mesh
        )
        cz = get_codewords(shortcodes=z, codebook=sg(c), global_mesh=self.global_mesh)
        cz = cz.astype(orig_dtype)
        vecs_hat = sg(cz) + st(vecs)  # exact by sterbenz's lemma
        if self.is_train:
            if self.head_type != "shga":
                losses_commit = jnp.sum(losses_commit, axis=1)  # sum over heads
                losses_codebook = jnp.sum(losses_codebook, axis=1)  # sum over heads
            l_commit = jnp.mean(losses_commit).astype(self.dtype)
            l_codebook = jnp.mean(losses_codebook).astype(self.dtype)
            metrics = None
        else:
            l_commit = jnp.zeros(dtype=self.dtype, shape=vecs.shape[:-1])
            l_codebook = jnp.zeros(dtype=self.dtype, shape=vecs.shape[:-1])
            metrics = None
        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
        )


class EMAVectorQuantizer(nn.Module):
    config: TransformerConfig
    global_mesh: jax.sharding.Mesh

    def setup(self):
        self.apply_config()
        self.c_sum = self.param(
            "codebook_sum",
            nn.with_partitioning(
                self.w_init, names=(None, None), mesh=self.global_mesh
            ),
            [self.n_code, self.d_k],
            self.param_dtype,
        )
        self.c_count = self.param(
            "codebook_count",
            nn.with_partitioning(
                jax.nn.initializers.ones, names=(None,), mesh=self.global_mesh
            ),
            [self.n_code],
            self.param_dtype,
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def _get_codebook(c_sum, c_count, dtype):
        c = jnp.divide(c_sum, jnp.clip(jnp.expand_dims(c_count, -1), a_min=0.01))
        return sg(c).astype(dtype)

    def get_codebook(self):
        return EMAVectorQuantizer._get_codebook(self.c_sum, self.c_count, self.dtype)

    @staticmethod
    def get_codebook_ema_targets(
        vecs,
        shortcodes,
        c_sum,
        c_count,
        c_gamma,
        n_device,
        n_block_per_update,
        global_mesh,
    ):
        n_code = c_sum.shape[0]
        dims = chex.Dimensions(
            B=vecs.shape[0],
            R=vecs.shape[1],
            C=vecs.shape[2],
            S=c_sum.shape[0],
            K=c_sum.shape[1],
            i=1,
        )
        vecs = sharding_constraint(vecs, global_mesh, ("data", None, None, None))
        shortcodes = sharding_constraint(shortcodes, global_mesh, ("data", None, None))
        c_sum = sharding_constraint(c_sum, global_mesh, (None, None))
        c_count = sharding_constraint(c_count, global_mesh, (None,))
        chex.assert_shape(vecs, dims["BRCK"])
        chex.assert_shape(shortcodes, dims["BRC"])
        chex.assert_shape(c_sum, dims["SK"])
        chex.assert_shape(c_count, dims["S"])

        assert dims["S"][0] == n_code
        g = c_gamma
        d = n_device
        p = n_block_per_update
        chex.assert_shape(d, dims["i"])
        chex.assert_shape(p, dims["i"])

        r = jax.nn.one_hot(shortcodes, num_classes=n_code, dtype=vecs.dtype)
        r = sharding_constraint(r, global_mesh, ("data", None, None, None))

        c_sum_hat = d * p * jnp.einsum("brcs,brck->sk", r, vecs)
        c_count_hat = d * p * jnp.einsum("brcs->s", r)
        c_sum_hat = sharding_constraint(c_sum_hat, global_mesh, (None, None))
        c_count_hat = sharding_constraint(c_count_hat, global_mesh, (None,))

        c_sum_tgt = (1 - g) * c_sum_hat + g * c_sum
        c_count_tgt = (1 - g) * c_count_hat + g * c_count
        c_sum_tgt = sharding_constraint(c_sum_tgt, global_mesh, (None, None))
        c_count_tgt = sharding_constraint(c_count_tgt, global_mesh, (None,))
        chex.assert_shape(c_sum_tgt, dims["SK"])
        chex.assert_shape(c_count_tgt, dims["S"])

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
        global_mesh,
    ):
        # the returned l_codebook gives correct updates
        # if gradients are averaged over data shards
        # and codebook optimizer is sgd with lr = 1.0.
        c_sum_tgt, c_count_tgt = EMAVectorQuantizer.get_codebook_ema_targets(
            vecs=vecs,
            shortcodes=shortcodes,
            c_sum=c_sum,
            c_count=c_count,
            c_gamma=c_gamma,
            n_device=n_device,
            n_block_per_update=n_block_per_update,
            global_mesh=global_mesh,
        )
        l_codebook_sum = jnp.sum(sg(jnp.subtract(c_sum, c_sum_tgt)) * st(c_sum))
        l_codebook_count = jnp.sum(sg(jnp.subtract(c_count, c_count_tgt)) * st(c_count))
        l_codebook = l_codebook_count + l_codebook_sum
        return l_codebook

    @staticmethod
    def get_quantization_metrics(vecs, vecs_hat, errs2, c_sum, c_count, dtype):
        # we'll call stop gradients in the return statement, so no need to call it now
        n_code = c_count.shape[0]
        eps, errmin, errmax, maskval = 1e-2, 0e1, 1e1, 1e30
        c_count = jnp.clip(c_count, a_min=eps)
        c = c_sum / c_count[..., None]  # SK
        c_norms = jnp.clip(jnp.linalg.norm(c, axis=-1), a_min=eps)  # S
        c_normed = c / c_norms[..., None]  # SK
        c_sims = jnp.einsum("sk,zk->sz", c_normed, c_normed)  # SS
        c_dists = jnp.linalg.norm(
            jnp.expand_dims(c, 1) - jnp.expand_dims(c, 0), axis=-1
        )  # SS
        vec_norms = jnp.clip(jnp.linalg.norm(vecs, axis=-1), a_min=eps)  # BL
        vec_hat_norms = jnp.clip(jnp.linalg.norm(vecs_hat, axis=-1), a_min=eps)  # BL
        errs = jnp.sqrt(errs2)  # BL
        relative_errs = jnp.clip(errs / vec_norms, errmin, errmax)  # BL
        probs = c_count / jnp.sum(c_count, axis=-1)[..., None]  # S
        c_thresh_oob = jnp.logical_or(c_count < 1.0, 1_000_000 < c_count)
        c_thresh_oob = c_thresh_oob.astype(jnp.float32)

        # elements will have shape [], [B], then we will tree map
        # to avg over heads/device batch items
        ones = jnp.ones([n_code, n_code], dtype=jnp.float32)
        up = jnp.triu(ones)  # upper triangular ones mask
        low = jnp.tril(ones, k=-1)  # strict lower triangular ones mask
        metrics = dict(
            c_sim_min=jnp.min(low * c_sims + maskval * up, axis=(0, 1)),  # []
            c_sim_mean=jnp.sum(low * c_sims, axis=(0, 1)) / jnp.sum(low, axis=(0, 1)),
            c_sim_max=jnp.max(low * c_sims - maskval * up, axis=(0, 1)),  # []
            c_dist_min=jnp.min(low * c_dists + maskval * up, axis=(0, 1)),  # []
            c_dist_mean=jnp.sum(low * c_dists, axis=(0, 1)) / jnp.sum(low, axis=(0, 1)),
            c_dist_max=jnp.max(low * c_dists - maskval * up, axis=(0, 1)),  # []
            c_norm_min=jnp.min(c_norms, axis=0),  # []
            c_norm_mean=jnp.mean(c_norms, axis=0),  # []
            c_norm_max=jnp.max(c_norms, axis=0),  # []
            c_usage_min=jnp.min(c_count, axis=0),  # []
            c_usage_mean=jnp.mean(c_count, axis=0),  # []
            c_usage_max=jnp.max(c_count, axis=0),  # []
            c_thresh_oob=jnp.sum(c_thresh_oob, axis=0),  # []
            c_entropy=jnp.sum(jsp.special.entr(probs), axis=-1),  # []
            vec_norm_mean=jnp.mean(vec_norms, axis=-1),  # [B]
            vec_hat_norm_mean=jnp.mean(vec_hat_norms, axis=-1),  # [B]
            relative_err_min=jnp.min(relative_errs, axis=-1),  # [B]
            relative_err_mean=jnp.mean(relative_errs, axis=-1),  # [B]
            relative_err_max=jnp.max(relative_errs, axis=-1),  # [B]
        )
        return jax.tree_util.tree_map(lambda x: jnp.mean(sg(x)).astype(dtype), metrics)

    def __call__(self, vecs):
        assert self.head_type == "shga", "Need to edit things for tp if using mha/mqa"
        orig_dtype = vecs.dtype
        vecs_hp = vecs.astype(jnp.float32)
        c = EMAVectorQuantizer._get_codebook(self.c_sum, self.c_count, jnp.float32)
        z, errs2 = get_shortcodes(
            vecs=vecs_hp, codebook=c, global_mesh=self.global_mesh
        )
        errs2 = errs2.astype(self.dtype)
        cz = get_codewords(shortcodes=z, codebook=c, global_mesh=self.global_mesh)
        cz = cz.astype(orig_dtype)
        vecs_hat = sg(cz) + st(vecs)  # exact by sterbenz's lemma
        if self.is_train:
            if self.head_type != "shga":
                errs2 = jnp.sum(errs2, axis=1)  # sum over heads
            l_commit = jnp.mean(errs2).astype(self.dtype)
            l_codebook = EMAVectorQuantizer.get_codebook_loss(
                vecs=vecs_hp,
                shortcodes=z,
                c_sum=self.c_sum,
                c_count=self.c_count,
                c_gamma=self.c_gamma,
                n_device=jnp.array([self.n_device]),
                n_block_per_update=jnp.array([self.sequence_len // self.block_len]),
                global_mesh=self.global_mesh,
            ).astype(self.dtype)
            # metrics = EMAVectorQuantizer.get_quantization_metrics(
            #    vecs=sg(vecs),
            #    vecs_hat=sg(vecs_hat),
            #    errs2=sg(errs2),
            #    c_sum=sg(self.c_sum),
            #    c_count=sg(self.c_count),
            #    dtype=self.dtype,
            # )
            metrics = None
        else:
            l_commit = jnp.zeros(dtype=self.dtype, shape=[])
            l_codebook = jnp.zeros(dtype=self.dtype, shape=[])
            metrics = None
        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            errs2=errs2,
            metrics=metrics,
        )
