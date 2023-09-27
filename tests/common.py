"""
Pytest fixtures. Due to the heterogeneity between tests, most fixtures are factories.
"""
import os

N_LOCAL_DEVICE = 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_LOCAL_DEVICE}"

import jax
import jax.numpy as jnp
import pytest
import dataclasses

from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import VQSpec
from transformer_vq.nn.attn import VQAttention
from transformer_vq.nn.model import Transformer

jax.config.update("jax_enable_x64", True)


WIDENINGS = [1]
DTYPES = [jnp.float32]
TOLERANCES = dict(atol=1e-5, rtol=1e-4)


def gen_len_tuples(order, t=12):
    # order is a string containing one or more of the following: {t, l, m}.
    # these correspond to sequence len, block len, and mem len, respectively.
    tuples = []
    for m in range(1, t):
        if t % m == 0:
            for ell in range(1, t):
                if t % ell == 0:
                    dict_ = dict(t=t, l=ell, m=m)
                    list_ = []
                    for letter in order:
                        list_.append(dict_[letter])
                    tuple_ = tuple(list_)
                    tuples.append(tuple_)
    tuples = list(set(tuples))
    return tuples


@pytest.fixture
def rng_fixture():
    def _rng(seed):
        return jax.random.PRNGKey(seed)

    return _rng


@pytest.fixture
def transformer_config_fixture():
    def _transformer_config(
        *,
        block_len,
        mem_len,
        agg_cache,
        widening,
        dtypes,
        is_train,
        n_vocab=10,
        c_gamma=0.99,
        no_emb=False,
        global_batch_size=None,  # set to none if not used
        sequence_len=None,  # set to none if not used
        update_len=None,  # set to none if not used
    ):
        return TransformerConfig.create(
            param_dtype=dtypes,
            dtype=dtypes,
            global_batch_size=global_batch_size,
            sequence_len=sequence_len,
            update_len=update_len,
            block_len=block_len,
            mem_len=mem_len,
            grad_thru_cache=True,
            agg_cache=agg_cache,
            d_model=4 * widening,
            d_k=4 * widening,
            d_v=8 * widening,
            d_ff=0,
            n_head=2,
            n_code=8,
            n_layer=2,
            n_vocab=n_vocab,
            pe_abs=True,
            pe_lam=10_000,
            p_dropemb=0.0,
            p_dropsin=0.0,
            p_dropres=0.0,
            p_droplyr=0.0,
            p_nucleus=0.8,  # used in sampling only
            c_beta=0.02,
            c_gamma=c_gamma,
            e_tie=True,
            e_preln=True,
            e_scale=1.0,
            is_train=is_train,
            no_emb=no_emb,
        )

    return _transformer_config


@pytest.fixture
def multidev_vq_spec_fixture():
    def _multidev_vq_spec(
        batch_size,
        block_len,
        n_device=N_LOCAL_DEVICE,
        n_local_device=N_LOCAL_DEVICE,
        n_update=1,
        n_block_per_update=1,
        update_id=0,
        block_id=0,
    ):
        assert batch_size % N_LOCAL_DEVICE == 0
        kwargs = dict(shape=[n_device, 1], dtype=jnp.int32)
        return VQSpec.create(
            n_device=jnp.full(fill_value=n_device, **kwargs),
            n_block_per_update=jnp.full(fill_value=n_block_per_update, **kwargs),
            loss_mask=jnp.ones(
                [N_LOCAL_DEVICE, batch_size // N_LOCAL_DEVICE, block_len], jnp.int32
            ),
        )

    return _multidev_vq_spec


@pytest.fixture
def basic_vq_spec_fixture():
    def _basic_vq_spec(
        batch_size,
        block_len,
        n_update=1,
        n_block_per_update=1,
        update_id=0,
        block_id=0,
    ):
        return VQSpec.create(
            n_device=jnp.array([1]),
            n_block_per_update=jnp.array([n_block_per_update]),
            loss_mask=jnp.ones([batch_size, block_len], jnp.int32),
        )

    return _basic_vq_spec


@pytest.fixture
def attn_fixture(rng_fixture, basic_vq_spec_fixture):
    def _attn(config: TransformerConfig):
        # for init, need to use initial state variant that doesn't depend on params,
        # so we override the config internally within this fixture for init.
        config = dict(dataclasses.asdict(config).items())  # make a copy
        config.update({"is_train": True})
        config = TransformerConfig.create(**config)
        cls = VQAttention
        if hasattr(config, "attn_type"):
            atp = config.attn_type
            if atp != "vq":
                raise ValueError(f"attention type {atp} not supported by fixture")
        present_len = config.block_len
        params = cls(config).init(
            dict(
                params=rng_fixture(1),
                ephemeral=rng_fixture(2),
                timeless=rng_fixture(3),
            ),
            state=cls.initial_state(config=config, batch_size=1),
            input_dict=dict(
                input_features=jax.random.normal(
                    rng_fixture(0), [1, present_len, config.d_model], dtype=config.dtype
                ),
                doc_ids=jnp.zeros([1, present_len], dtype=jnp.int32),
                vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=present_len),
            ),
        )["params"]
        return cls, params

    return _attn


@pytest.fixture
def transformer_fixture(rng_fixture, basic_vq_spec_fixture):
    def _transformer(config):
        # for init, need to use initial state variant that doesn't depend on params,
        # so we override the config internally within this fixture for init.
        config = dict(dataclasses.asdict(config).items())  # make a copy
        config.update({"is_train": True})
        config = TransformerConfig.create(**config)
        present_len = config.block_len
        rngs = dict(
            params=rng_fixture(1),
            ephemeral=rng_fixture(2),
            timeless=rng_fixture(3),
        )
        if config.no_emb:
            inputs = jax.random.normal(
                key=rng_fixture(0),
                shape=[1, present_len, config.d_model],
                dtype=config.dtype,
            )
        else:
            inputs = jax.random.randint(
                key=rng_fixture(0),
                minval=0,
                maxval=config.n_vocab,
                shape=[1, present_len],
            )
        doc_ids = jnp.zeros([1, present_len], dtype=jnp.int32)
        state = Transformer.initial_state(config=config, batch_size=1)
        params = Transformer(config).init(
            rngs,
            inputs=inputs,
            doc_ids=doc_ids,
            state=state,
            vq_spec=basic_vq_spec_fixture(batch_size=1, block_len=present_len),
        )["params"]
        return Transformer, params

    return _transformer
