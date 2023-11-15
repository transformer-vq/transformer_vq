import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_vq.nn.attn1 import VQAttentionOld
from transformer_vq.nn.qkv import QKVGProducer

# noreorder
from tests.common import WIDENINGS
from tests.common import ATTN_TYPES_OLD_UNWRAPPED
from tests.common import gen_len_tuples
from tests.common import gen_type_tuples
from tests.common import attn_fixture
from tests.common import transformer_config_fixture
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
    biases = VQAttentionOld.get_agg_biases(counts)
    np.testing.assert_allclose(jnp.exp(biases), counts)

    attn_scores = jax.random.normal(rng_fixture(0), shape=[timesteps])
    biased_attn_scores = attn_scores + biases
    np.testing.assert_allclose(
        jnp.exp(biased_attn_scores), counts * jnp.exp(attn_scores)
    )


@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize(
    "attn_type,head_type", gen_type_tuples(attn_type=ATTN_TYPES_OLD_UNWRAPPED)
)
def test_attn_jacobian_blockonly(
    rng_fixture,
    attn_fixture,
    transformer_config_fixture,
    widening,
    attn_type,
    head_type,
    is_train=True,
    sequence_len=12,
    dtype=jnp.float32,
):
    config = transformer_config_fixture(
        sequence_len=sequence_len,
        block_len=sequence_len,
        widening=widening,
        dtypes=dtype,
        attn_type=attn_type,
        head_type=head_type,
        is_train=is_train,
    )
    cls, params = attn_fixture(config)
    initial_state = cls.initial_state(config, batch_size=1)

    def call_fn(x):
        # takes a slice in R^L of the attn inputs and returns a slice in R^L of output
        x = jnp.pad(x[None, ..., None], ((0, 0), (0, 0), (0, config.d_model - 1)))
        _, attn_output_dict = cls(config).apply(
            {"params": params},
            state=initial_state,
            x=x,
            rngs=dict(ephemeral=rng_fixture(1), timeless=rng_fixture(2)),
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
