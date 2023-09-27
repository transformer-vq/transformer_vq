import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_vq.nn.attn import VQAttention

# noreorder
from tests.common import DTYPES
from tests.common import WIDENINGS
from tests.common import gen_len_tuples
from tests.common import attn_fixture
from tests.common import transformer_config_fixture
from tests.common import basic_vq_spec_fixture
from tests.common import rng_fixture


def test_jax_nn_one_hot():
    np.testing.assert_allclose(
        actual=jax.nn.one_hot(jnp.array(0), num_classes=3, axis=-1, dtype=jnp.float32),
        desired=jnp.array([1, 0, 0], dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        actual=jax.nn.one_hot(jnp.array(1), num_classes=3, axis=-1, dtype=jnp.float32),
        desired=jnp.array([0, 1, 0], dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        actual=jax.nn.one_hot(jnp.array(2), num_classes=3, axis=-1, dtype=jnp.float32),
        desired=jnp.array([0, 0, 1], dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        actual=jax.nn.one_hot(jnp.array(3), num_classes=3, axis=-1, dtype=jnp.float32),
        desired=jnp.array([0, 0, 0], dtype=jnp.float32),
    )


def test_jnp_take_along_axis():
    c = jnp.reshape(jnp.arange(30), [2, 3, 5])
    z = jnp.reshape(jnp.arange(0, 5), (1, 1, 5))
    cz_actual = jnp.take_along_axis(c, z, axis=2)
    cz_expected = c
    np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)

    c = jnp.reshape(jnp.arange(30), [2, 3, 5])
    z = jnp.reshape(jnp.arange(0, 5), (1, 1, 5))[:, :, 0:4]
    cz_actual = jnp.take_along_axis(c, z, axis=2)
    cz_expected = c[:, :, 0:4]
    np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)

    c = jnp.reshape(jnp.arange(210), [2, 3, 5, 7])
    z = jnp.reshape(jnp.arange(0, 5), (1, 1, 5))[:, :, 0:4]
    cz_actual = jnp.take_along_axis(c, jnp.expand_dims(z, -1), axis=2)
    cz_expected = c[:, :, 0:4, :]
    np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)

    c = jnp.reshape(jnp.arange(210), [2, 3, 5, 7])
    z = jnp.reshape(jnp.remainder(jnp.arange(0, 10), jnp.array(5)), (1, 1, 10))
    cz_actual = jnp.take_along_axis(c, jnp.expand_dims(z, -1), axis=2)
    cz_expected = jnp.concatenate([c, c], axis=2)
    np.testing.assert_allclose(actual=cz_actual, desired=cz_expected)


def test_get_agg_biases(rng_fixture):
    timesteps = 1024

    counts = jnp.arange(timesteps)
    biases = VQAttention.get_agg_biases(counts)
    np.testing.assert_allclose(jnp.exp(biases), counts)

    attn_scores = jax.random.normal(rng_fixture(0), shape=[timesteps])
    biased_attn_scores = attn_scores + biases
    np.testing.assert_allclose(
        jnp.exp(biased_attn_scores), counts * jnp.exp(attn_scores)
    )


@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("agg_cache", [True, False])
def test_attn_jacobian(
    rng_fixture,
    attn_fixture,
    transformer_config_fixture,
    basic_vq_spec_fixture,
    widening,
    agg_cache,
    is_train=True,
    sequence_len=10,
    dtype=jnp.float32,
):
    config = transformer_config_fixture(
        block_len=sequence_len,
        mem_len=sequence_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    cls, params = attn_fixture(config)
    initial_state = cls.initial_state(config, batch_size=1)

    def call_fn(x):
        # takes a slice in R^L of the attn inputs and returns a slice in R^L of output
        _, attn_output_dict = cls(config).apply(
            {"params": params},
            state=initial_state,
            input_dict=dict(
                input_features=jnp.pad(
                    x[None, ..., None], ((0, 0), (0, 0), (0, config.d_model - 1))
                ),
                doc_ids=jnp.ones([1, sequence_len], dtype=jnp.int32),
                vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=sequence_len),
            ),
            rngs=dict(
                ephemeral=rng_fixture(1),
                timeless=rng_fixture(2),
            ),
        )
        return attn_output_dict["res"][0, :, 0]

    inputs = jax.random.normal(rng_fixture(0), [sequence_len], dtype=config.dtype)
    jac = jax.jacfwd(call_fn)(inputs)
    print(jac)
    # check that outputs do not have nonzero grad wrt future inputs
    np.testing.assert_allclose(actual=jac, desired=jnp.tril(jac), atol=1e-9, rtol=1e-9)
    with pytest.raises(AssertionError):
        # check that outputs *do* have nonzero grad wrt past/present inputs:
        # if they only have zero grad, the assertion will pass, and the test will fail
        np.testing.assert_allclose(
            actual=jac, desired=jnp.zeros_like(jac), atol=1e-9, rtol=1e-9
        )


@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("agg_cache", [True, False])
@pytest.mark.parametrize("sequence_len,block_len,mem_len", gen_len_tuples("tlm", 12))
def test_attn_forward_consistency(
    rng_fixture,
    attn_fixture,
    transformer_config_fixture,
    basic_vq_spec_fixture,
    widening,
    agg_cache,
    sequence_len,
    block_len,
    mem_len,
    dtype=jnp.float32,
    is_train=True,
):
    config_monoblock = transformer_config_fixture(
        block_len=sequence_len,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    bsz = 3

    cls, params = attn_fixture(config_monoblock)
    initial_state = cls.initial_state(config=config_monoblock, batch_size=bsz)

    inputs_shape = [bsz, sequence_len, config_monoblock.d_model]
    inputs = jax.random.normal(rng_fixture(0), inputs_shape, dtype=dtype)
    _, output_dict = cls(config_monoblock).apply(
        {"params": params},
        state=initial_state,
        input_dict=dict(
            input_features=inputs,
            doc_ids=jnp.ones([bsz, sequence_len], dtype=jnp.int32),
            vq_spec=basic_vq_spec_fixture(
                batch_size=bsz,
                block_len=config_monoblock.block_len,
            ),
        ),
        rngs=dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2)),
    )
    o_expected = output_dict["res"]

    config_multiblock = transformer_config_fixture(
        block_len=block_len,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    o_actual = []
    state_p = cls.initial_state(
        config=config_multiblock,
        batch_size=bsz,
    )
    for p in range(sequence_len // block_len):
        state_p, output_dict_p = cls(config_multiblock).apply(
            {"params": params},
            state=state_p,
            input_dict=dict(
                input_features=inputs[:, p * block_len : (p + 1) * block_len, :],
                doc_ids=jnp.ones([bsz, block_len], dtype=jnp.int32),
                vq_spec=basic_vq_spec_fixture(
                    batch_size=bsz,
                    block_len=config_multiblock.block_len,
                ),
            ),
            rngs=dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2)),
        )
        o_actual.append(output_dict_p["res"])
    o_actual = jnp.concatenate(o_actual, axis=1)
    assert o_expected.shape[1] == sequence_len
    assert o_actual.shape == o_expected.shape
    if dtype is jnp.float64:
        np.testing.assert_allclose(
            actual=o_actual,
            desired=o_expected,
            atol=1e-5,
            rtol=1e-4,
        )
    if dtype is jnp.float32:
        np.testing.assert_allclose(
            actual=o_actual,
            desired=o_expected,
            atol=5e-4,
            rtol=5e-3,
        )

    """
    # uncomment to inspect which timesteps are bad:
    for t in range(sequence_len):
        print(t)
        if dtype is jnp.float64:
            np.testing.assert_allclose(
                actual=o_actual[:, t, :],
                desired=o_expected[:, t, :],
                atol=1e-5,
                rtol=1e-4,
            )
        if dtype is jnp.float32:
            np.testing.assert_allclose(
                actual=o_actual[:, t, :],
                desired=o_expected[:, t, :],
                atol=5e-4,
                rtol=5e-3,
            )
        """


@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("agg_cache", [True, False])
@pytest.mark.parametrize("sequence_len,mem_len", gen_len_tuples("tm", 12))
def test_attn_backward_consistency(
    rng_fixture,
    attn_fixture,
    transformer_config_fixture,
    basic_vq_spec_fixture,
    widening,
    agg_cache,
    sequence_len,
    mem_len,
    is_train=True,
    dtype=jnp.float32,
):
    bsz = 1
    config_monoblock = transformer_config_fixture(
        block_len=sequence_len,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    cls, params = attn_fixture(config_monoblock)
    monoblock_initial_state = cls.initial_state(
        config=config_monoblock,
        batch_size=bsz,
    )

    prefix = [bsz, config_monoblock.n_head, sequence_len]
    d_k = config_monoblock.d_k
    d_v = config_monoblock.d_v
    q = jax.random.normal(rng_fixture(1), [*prefix, d_k], dtype=config_monoblock.dtype)
    q_slice = q[0, -1, :, -1]
    k = jax.random.normal(rng_fixture(2), [*prefix, d_k], dtype=config_monoblock.dtype)
    v = jax.random.normal(rng_fixture(3), [*prefix, d_v], dtype=config_monoblock.dtype)
    pad_spec = (
        (0, 0),
        (config_monoblock.n_head - 1, 0),
        (0, 0),
        (config_monoblock.d_k - 1, 0),
    )
    vq_spec_full_seq = basic_vq_spec_fixture(batch_size=bsz, block_len=sequence_len)
    rngs = dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2))

    def _get_expected_second_block_jaco_wrt_q():
        # expected second block jacobian, computed from full attn over sequence
        jac_fn = jax.jacobian(
            lambda x: cls(config_monoblock).apply(
                {"params": params},
                present_q=jnp.pad(x[None, None, ..., None], pad_spec),
                present_k=k,
                present_v=v,
                present_doc_ids=jnp.ones([bsz, sequence_len], dtype=jnp.int32),
                state=monoblock_initial_state,
                vq_spec=vq_spec_full_seq,
                rngs=rngs,
                method=cls.attn,
            )["attn_out"][0, :, -1]
        )
        return jac_fn(q_slice)[sequence_len // 2 :, sequence_len // 2 :]

    config_multiblock = transformer_config_fixture(
        block_len=sequence_len // 2,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    multiblock_initial_state = cls.initial_state(
        config=config_multiblock,
        batch_size=bsz,
    )

    vq_spec_half_seq = basic_vq_spec_fixture(
        batch_size=bsz, block_len=sequence_len // 2
    )
    attn_outputs_midway = cls(config_multiblock).apply(
        {"params": params},
        present_q=jnp.pad(
            q_slice[0 : (sequence_len // 2)][None, None, ..., None],
            pad_spec,
        ),
        present_k=k[..., 0 : (sequence_len // 2), :],
        present_v=v[..., 0 : (sequence_len // 2), :],
        present_doc_ids=jnp.ones([bsz, sequence_len // 2], dtype=jnp.int32),
        state=multiblock_initial_state,
        vq_spec=vq_spec_half_seq,
        rngs=rngs,
        method=cls.attn,
    )
    update_kwargs = dict(
        recent_z=attn_outputs_midway["recent_z"],
        recent_k_hat=attn_outputs_midway["recent_k_hat"],
        recent_v=attn_outputs_midway["recent_v"],
        recent_doc_ids=attn_outputs_midway["recent_doc_ids"],
    )
    midway_state = cls(config_multiblock).apply(
        {"params": params},
        **update_kwargs,
        state=multiblock_initial_state,
        method=cls.update_state,
    )

    def _get_actual_second_block_jaco_wrt_q():
        # actual second block jacobian
        jac_fn = jax.jacobian(
            lambda x: cls(config_multiblock).apply(
                {"params": params},
                present_q=jnp.pad(x[None, None, ..., None], pad_spec),
                present_k=k[..., sequence_len // 2 :, :],
                present_v=v[..., sequence_len // 2 :, :],
                present_doc_ids=jnp.ones([bsz, sequence_len // 2], dtype=jnp.int32),
                state=midway_state,
                vq_spec=vq_spec_half_seq,
                rngs=rngs,
                method=cls.attn,
            )["attn_out"][0, :, -1]
        )
        return jac_fn(q_slice[sequence_len // 2 :])

    jac_actual = _get_actual_second_block_jaco_wrt_q()
    jac_expected = _get_expected_second_block_jaco_wrt_q()
    np.testing.assert_allclose(
        actual=jac_actual, desired=jac_expected, atol=1e-5, rtol=1e-4
    )
