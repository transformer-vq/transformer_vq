import functools
import os

N_LOCAL_DEVICE = 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_LOCAL_DEVICE}"

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import jax_utils
from flax.training import common_utils

from transformer_vq.nn.vq import VectorQuantizer, get_shortcodes, get_codewords
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.attn1 import QKVGProducer

# noreorder
from tests.common import transformer_config_fixture, N_LOCAL_DEVICE
from tests.common import rng_fixture
from tests.common import TOLERANCES

jax.config.update("jax_enable_x64", True)


def test_num_devices():
    assert jax.device_count() == N_LOCAL_DEVICE
    assert jax.local_device_count() == N_LOCAL_DEVICE


@pytest.fixture
def quantizer_fixture(rng_fixture):
    def _make_quantizer(cls, config: TransformerConfig):
        n_head = QKVGProducer.get_n_kv(config)
        inputs_shape = [1, n_head, 1, 8, config.d_k]
        inputs = jax.random.normal(rng_fixture(0), inputs_shape, dtype=jnp.float32)
        rngs = dict(
            params=rng_fixture(1),
            ephemeral=rng_fixture(2),
            timeless=rng_fixture(3),
        )
        params = cls(config=config, n_head=n_head).init(
            rngs,
            vecs=inputs.astype(config.dtype),
        )["params"]
        return cls, params

    return _make_quantizer


def test_get_shortcodes():
    h, l, s, d = 3, 5, 7, 11
    codebook = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    codebook = jnp.arange(1, h + 1).reshape([h, 1, 1]) * codebook.reshape([1, s, d])
    vecs = codebook[:, -l:, :].reshape(1, h, 1, l, d)
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=codebook)
    np.testing.assert_allclose(
        actual=shortcodes,
        desired=jnp.tile(jnp.arange(s - l, s).reshape([1, 1, 1, l]), reps=[1, h, 1, 1]),
    )


def test_get_codewords():
    h, l, s, d = 3, 5, 7, 11
    codebook = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    codebook = jnp.arange(1, h + 1).reshape([h, 1, 1]) * codebook.reshape([1, s, d])
    vecs = codebook[:, -l:, :].reshape(1, h, 1, l, d)
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=codebook)
    codewords = get_codewords(shortcodes=shortcodes, codebook=codebook)
    np.testing.assert_allclose(actual=codewords, desired=vecs)


def test_get_codebook_ema_targets():
    h, l, s, d = 3, 5, 7, 11
    c_gamma = 0.99
    ones = jnp.ones([h, s], dtype=jnp.float64)
    c_count = ones
    c_sum = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    c_sum = jnp.arange(1, h + 1).reshape([h, 1, 1]) * c_sum.reshape([1, s, d])
    vecs = c_sum[:, -l:, :].reshape(1, h, 1, l, d)
    vecs = vecs + 0.1 * jnp.ones_like(vecs)
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=c_sum)
    np.testing.assert_allclose(
        actual=shortcodes,
        desired=jnp.tile(jnp.arange(s - l, s).reshape([1, 1, 1, l]), reps=[1, h, 1, 1]),
    )
    c_sum_tgt_expected = c_gamma * c_sum + (1 - c_gamma) * jnp.pad(
        vecs[0, :, 0, ...], ((0, 0), (s - l, 0), (0, 0))
    )
    c_count_tgt_expected = c_gamma * c_count + (1 - c_gamma) * jnp.pad(
        ones[:, -l:], ((0, 0), (s - l, 0))
    )
    c_sum_tgt_actual, c_count_tgt_actual = VectorQuantizer.get_codebook_ema_targets(
        vecs=vecs,
        shortcodes=shortcodes,
        c_sum=c_sum,
        c_count=c_count,
        c_gamma=c_gamma,
        n_device=jnp.array([1]),
        n_block_per_update=jnp.array([1]),
    )
    np.testing.assert_allclose(actual=c_sum_tgt_actual, desired=c_sum_tgt_expected)
    np.testing.assert_allclose(actual=c_count_tgt_actual, desired=c_count_tgt_expected)


def test_get_codebook_loss_ema(rng_fixture):
    h, l, s, d = 3, 22, 11, 41
    c_gamma = 0.99
    c_sum = jax.nn.one_hot(jnp.arange(s), num_classes=d)
    c_sum = jnp.arange(1, h + 1).reshape([h, 1, 1]) * c_sum.reshape([1, s, d])
    c_count = 2 * jnp.ones([h, s], dtype=jnp.float64)
    params = dict(c_sum=c_sum, c_count=c_count)
    vecs = jax.random.normal(rng_fixture(0), shape=[1, h, 1, l, d], dtype=jnp.float64)
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=c_sum)

    # make expected updates
    r = jax.nn.one_hot(shortcodes, num_classes=s, dtype=jnp.float64)
    c_sum_hat = jnp.einsum("bhrcs,bhrck->hsk", r, vecs)
    c_count_hat = jnp.einsum("bhrcs->hs", r)
    c_sum_new_expected = c_gamma * c_sum + (1 - c_gamma) * c_sum_hat
    c_count_new_expected = c_gamma * c_count + (1 - c_gamma) * c_count_hat

    def loss_fn(params_, vecs_, shortcodes_, n_device_, n_block_per_update_):
        return VectorQuantizer.get_codebook_loss(
            vecs=vecs_,
            shortcodes=shortcodes_,
            c_sum=params_["c_sum"],
            c_count=params_["c_count"],
            c_gamma=c_gamma,
            n_device=jnp.array([n_device_]),
            n_block_per_update=jnp.array([n_block_per_update_]),
        )

    # single block per update
    grads = jax.grad(loss_fn)(
        params,
        vecs_=vecs,
        shortcodes_=shortcodes,
        n_device_=1,
        n_block_per_update_=1,
    )
    c_sum_new_actual = c_sum - grads["c_sum"]
    c_count_new_actual = c_count - grads["c_count"]
    np.testing.assert_allclose(actual=c_sum_new_actual, desired=c_sum_new_expected)
    np.testing.assert_allclose(actual=c_count_new_actual, desired=c_count_new_expected)

    # multiple blocks per update, updates avg grads over blocks
    grads_block0 = jax.grad(loss_fn)(
        params,
        vecs_=vecs[..., : l // 2, :],
        shortcodes_=shortcodes[..., : l // 2],
        n_device_=1,
        n_block_per_update_=2,
    )
    grads_block1 = jax.grad(loss_fn)(
        params,
        vecs_=vecs[..., l // 2 :, :],
        shortcodes_=shortcodes[..., l // 2 :],
        n_device_=1,
        n_block_per_update_=2,
    )
    grads = jax.tree_util.tree_map(
        lambda a, b: 0.5 * (a + b), grads_block0, grads_block1
    )
    c_sum_new_actual = c_sum - grads["c_sum"]
    c_count_new_actual = c_count - grads["c_count"]
    np.testing.assert_allclose(actual=c_sum_new_actual, desired=c_sum_new_expected)
    np.testing.assert_allclose(actual=c_count_new_actual, desired=c_count_new_expected)


def test_get_codebook_loss_ema_multidev(rng_fixture):
    assert jax.device_count() == N_LOCAL_DEVICE
    assert jax.local_device_count() == N_LOCAL_DEVICE
    b = 3 * jax.local_device_count()
    h, l, s, d = 3, 22, 11, 41
    c_gamma = 0.99
    c_sum = jax.random.normal(rng_fixture(0), shape=[h, s, d], dtype=jnp.float64)
    c_count = 2.0 * jnp.ones([h, s], dtype=jnp.float64)
    params = dict(c_sum=c_sum, c_count=c_count)
    vecs = jax.random.normal(rng_fixture(1), shape=[b, h, 1, l, d], dtype=jnp.float64)
    shortcodes, _ = get_shortcodes(vecs=vecs, codebook=c_sum)

    # make expected updates
    r = jax.nn.one_hot(shortcodes, num_classes=s, dtype=jnp.float64)
    c_sum_hat = jnp.einsum("bhrcs,bhrck->hsk", r, vecs)
    c_count_hat = jnp.einsum("bhrcs->hs", r)
    c_sum_new_expected = c_gamma * c_sum + (1 - c_gamma) * c_sum_hat
    c_count_new_expected = c_gamma * c_count + (1 - c_gamma) * c_count_hat

    @functools.partial(jax.pmap, axis_name="devices")
    def grad_fn(params_, vecs_, shortcodes_, n_device_, n_block_per_update_):
        def loss_fn(ps):
            return VectorQuantizer.get_codebook_loss(
                vecs=vecs_,
                shortcodes=shortcodes_,
                c_sum=ps["c_sum"],
                c_count=ps["c_count"],
                c_gamma=c_gamma,
                n_device=n_device_,
                n_block_per_update=n_block_per_update_,
            )

        grads_ = jax.grad(loss_fn)(params_)
        return jax.lax.pmean(grads_, axis_name="devices")

    # single block per update
    grads = grad_fn(
        jax_utils.replicate(params),
        vecs_=common_utils.shard(vecs),
        shortcodes_=common_utils.shard(shortcodes),
        n_device_=jnp.full(fill_value=N_LOCAL_DEVICE, shape=[N_LOCAL_DEVICE, 1]),
        n_block_per_update_=jnp.full(fill_value=1, shape=[N_LOCAL_DEVICE, 1]),
    )
    grads = jax_utils.unreplicate(grads)
    c_sum_new_actual = c_sum - grads["c_sum"]
    c_count_new_actual = c_count - grads["c_count"]
    np.testing.assert_allclose(
        actual=c_sum_new_actual, desired=c_sum_new_expected, **TOLERANCES
    )
    np.testing.assert_allclose(
        actual=c_count_new_actual, desired=c_count_new_expected, **TOLERANCES
    )

    # multiple blocks per update
    grads_block0 = grad_fn(
        jax_utils.replicate(params),
        vecs_=common_utils.shard(vecs[..., : l // 2, :]),
        shortcodes_=common_utils.shard(shortcodes[..., : l // 2]),
        n_device_=jnp.full(fill_value=N_LOCAL_DEVICE, shape=[N_LOCAL_DEVICE, 1]),
        n_block_per_update_=jnp.full(fill_value=2, shape=[N_LOCAL_DEVICE, 1]),
    )
    grads_block0 = jax_utils.unreplicate(grads_block0)
    grads_block1 = grad_fn(
        jax_utils.replicate(params),
        vecs_=common_utils.shard(vecs[..., l // 2 :, :]),
        shortcodes_=common_utils.shard(shortcodes[..., l // 2 :]),
        n_device_=jnp.full(fill_value=N_LOCAL_DEVICE, shape=[N_LOCAL_DEVICE, 1]),
        n_block_per_update_=jnp.full(fill_value=2, shape=[N_LOCAL_DEVICE, 1]),
    )
    grads_block1 = jax_utils.unreplicate(grads_block1)
    grads = jax.tree_util.tree_map(
        lambda v1, v2: 0.5 * (v1 + v2), grads_block0, grads_block1
    )
    c_sum_new_actual = c_sum - grads["c_sum"]
    c_count_new_actual = c_count - grads["c_count"]
    np.testing.assert_allclose(
        actual=c_sum_new_actual, desired=c_sum_new_expected, **TOLERANCES
    )
    np.testing.assert_allclose(
        actual=c_count_new_actual, desired=c_count_new_expected, **TOLERANCES
    )


def test_vector_quantizer_call(
    rng_fixture, quantizer_fixture, transformer_config_fixture
):
    config = transformer_config_fixture(
        sequence_len=120,
        block_len=30,
        widening=1,
        dtypes=jnp.float32,
        is_train=True,
    )
    cls = VectorQuantizer
    _, params = quantizer_fixture(cls, config)

    b = 5
    l = config.block_len
    h = QKVGProducer.get_n_kv(config)
    d = config.d_k
    vecs = jax.random.normal(rng_fixture(1), shape=[b, h, 4, l, d], dtype=jnp.float64)
    rngs = dict(
        ephemeral=rng_fixture(2),
        timeless=rng_fixture(3),
    )

    def get_commit_loss(vecs_, params_):
        return cls(config, n_head=h).apply(
            {"params": params_},
            vecs=vecs_,
            rngs=rngs,
        )["l_commit"]

    def get_codebook_loss(params_, vecs_):
        return cls(config, n_head=h).apply(
            {"params": params_},
            vecs=vecs_,
            rngs=rngs,
        )["l_codebook"]

    grad = jax.grad(get_commit_loss)(vecs, params_=params)
    with pytest.raises(AssertionError):
        for leaf in jax.tree_util.tree_leaves(grad):
            np.testing.assert_allclose(actual=leaf, desired=jnp.zeros_like(leaf))

    grad = jax.grad(get_codebook_loss)(params, vecs_=vecs)
    with pytest.raises(AssertionError):
        for leaf in jax.tree_util.tree_leaves(grad):
            np.testing.assert_allclose(actual=leaf, desired=jnp.zeros_like(leaf))
