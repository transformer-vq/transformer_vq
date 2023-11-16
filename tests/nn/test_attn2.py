import flax.core.frozen_dict as frozen_dict
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# noreorder
from tests.common import WIDENINGS, HEAD_TYPES, REDUCTION_TYPES, TOLERANCES
from tests.common import gen_len_tuples
from tests.common import gen_type_tuples
from tests.common import attn_fixture
from tests.common import transformer_config_fixture
from tests.common import rng_fixture


@pytest.mark.parametrize("widening", WIDENINGS)
@pytest.mark.parametrize("attn_type,head_type,reduction_type", gen_type_tuples())
def test_attn_jacobian_fullseq(
    rng_fixture,
    attn_fixture,
    transformer_config_fixture,
    widening,
    attn_type,
    head_type,
    reduction_type,
    is_train=True,
    sequence_len=12,
    dtype=jnp.float32,
):
    config = transformer_config_fixture(
        sequence_len=sequence_len,
        block_len=sequence_len // 4,
        widening=widening,
        dtypes=dtype,
        attn_type=attn_type,
        head_type=head_type,
        reduction_type=reduction_type,
        is_train=is_train,
    )
    attn_cls, params = attn_fixture(config)

    def call_fn(x):
        # takes a slice in R^T of the attn inputs and returns a slice in R^T of output
        x = jnp.pad(x[None, ..., None], ((0, 0), (0, 0), (0, config.d_model - 1)))
        x = attn_cls.pre_reshape(x, config)
        x = attn_cls(config).apply(
            {"params": params},
            x=x,
            rngs=dict(ephemeral=rng_fixture(0), timeless=rng_fixture(1)),
        )["res"]
        x = attn_cls.post_reshape(x, config)
        return x[0, :, 0]

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
@pytest.mark.parametrize("head_type", HEAD_TYPES)
@pytest.mark.parametrize("reduction_type", REDUCTION_TYPES)
def test_vq_attn_impl_consistency(
    rng_fixture,
    attn_fixture,
    transformer_config_fixture,
    widening,
    head_type,
    reduction_type,
    sequence_len=120,
    dtype=jnp.float32,
):
    block_len = sequence_len // 12
    config_common = dict(
        sequence_len=sequence_len,
        block_len=block_len,
        widening=widening,
        dtypes=dtype,
        head_type=head_type,
        reduction_type=reduction_type,
        is_train=True,
    )
    config_new = transformer_config_fixture(**config_common, attn_type="vq")
    config_old = transformer_config_fixture(**config_common, attn_type="vq_old")
    attn_cls_new, params_new = attn_fixture(config_new)
    attn_cls_old, _ = attn_fixture(config_old)

    # force same params for both
    params_new_unfrozen = params_new.unfreeze()
    params_new_unfrozen_wrapped = {"ScanVQAttentionOld_0": params_new_unfrozen}
    params_old_refrozen_rewrapped = frozen_dict.FrozenDict(params_new_unfrozen_wrapped)
    params_old = params_old_refrozen_rewrapped

    # inputs
    inputs = jax.random.normal(
        rng_fixture(0), [1, sequence_len, config_new.d_model], dtype=config_new.dtype
    )

    # forward
    out_new = attn_cls_new(config_new).apply(
        {"params": params_new},
        x=attn_cls_new.pre_reshape(inputs, config_new),
        rngs=dict(ephemeral=rng_fixture(0), timeless=rng_fixture(1)),
    )
    res_new = attn_cls_new.post_reshape(out_new["res"], config_new)

    out_old = attn_cls_old(config_old).apply(
        {"params": params_old},
        x=attn_cls_old.pre_reshape(inputs, config_old),
        rngs=dict(ephemeral=rng_fixture(0), timeless=rng_fixture(1)),
    )
    res_old = attn_cls_new.post_reshape(out_old["res"], config_old)

    assert out_new.keys() == out_old.keys()
    for i in range(sequence_len // block_len):
        np.testing.assert_allclose(
            actual=res_new[:, i * block_len : (i + 1) * block_len, :],
            desired=res_old[:, i * block_len : (i + 1) * block_len, :],
            **TOLERANCES,
        )

    np.testing.assert_allclose(
        actual=out_new["l_commit"],
        desired=out_old["l_commit"],
        **TOLERANCES,
    )
