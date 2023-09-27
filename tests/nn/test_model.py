import jax
import jax.numpy as jnp
import numpy as np
import pytest

# noreorder
from tests.common import WIDENINGS
from tests.common import gen_len_tuples
from tests.common import transformer_fixture
from tests.common import transformer_config_fixture
from tests.common import basic_vq_spec_fixture
from tests.common import rng_fixture


@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("agg_cache", [True, False])
@pytest.mark.parametrize("sequence_len,block_len,mem_len", gen_len_tuples("tlm", 12))
def test_model_jacobian(
    rng_fixture,
    transformer_fixture,
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
    config = transformer_config_fixture(
        block_len=block_len,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
        no_emb=True,
    )
    cls, params = transformer_fixture(config)
    initial_state = cls.initial_state(config, batch_size=1)

    def call_fn(x):
        # takes a slice in R^L of the attn inputs and returns a slice in R^L of output
        transformer_output_dict = cls(config).apply(
            {"params": params},
            state=initial_state,
            inputs=jnp.pad(
                x[None, ..., None], ((0, 0), (0, 0), (0, config.d_model - 1))
            ),
            doc_ids=jnp.ones([1, sequence_len], dtype=jnp.int32),
            vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=sequence_len),
            rngs=dict(
                ephemeral=rng_fixture(1),
                timeless=rng_fixture(2),
            ),
        )
        return transformer_output_dict["logprobs"][0, :, 0]

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
def test_model_forward_consistency(
    rng_fixture,
    transformer_fixture,
    transformer_config_fixture,
    basic_vq_spec_fixture,
    widening,
    agg_cache,
    sequence_len,
    block_len,
    mem_len,
    is_train=True,
    dtype=jnp.float32,
):
    bsz = 1
    config = transformer_config_fixture(
        block_len=block_len,
        mem_len=mem_len,
        agg_cache=agg_cache,
        widening=widening,
        dtypes=dtype,
        is_train=is_train,
    )
    cls, params = transformer_fixture(config)
    inputs = jax.random.randint(
        rng_fixture(0), minval=0, maxval=config.n_vocab, shape=[bsz, sequence_len]
    )
    initial_state = cls.initial_state(config=config, batch_size=bsz)
    rngs = dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2))

    o_expected = (
        cls(config)
        .apply(
            {"params": params},
            inputs=inputs,
            doc_ids=jnp.ones([bsz, sequence_len], dtype=jnp.int32),
            state=initial_state,
            vq_spec=basic_vq_spec_fixture(batch_size=bsz, block_len=sequence_len),
            rngs=rngs,
        )
        .get("logprobs")
    )
    o_actual = []
    state_p = cls.initial_state(config=config, batch_size=bsz)
    for p in range(sequence_len // block_len):
        slice_ = slice(p * block_len, (p + 1) * block_len)
        inputs_p = inputs[:, slice_]
        results = cls(config).apply(
            {"params": params},
            inputs=inputs_p,
            doc_ids=jnp.ones([bsz, block_len], dtype=jnp.int32),
            state=state_p,
            vq_spec=basic_vq_spec_fixture(batch_size=bsz, block_len=block_len),
            rngs=rngs,
        )
        o_p = results.get("logprobs")
        state_p = results.get("attn_state")
        print(p)
        print(state_p)
        o_actual.append(o_p)
    o_actual = jnp.concatenate(o_actual, axis=1)
    assert o_expected.shape[1] == sequence_len
    assert o_actual.shape == o_expected.shape
    np.testing.assert_allclose(
        actual=o_actual, desired=o_expected, atol=1e-5, rtol=1e-3
    )
