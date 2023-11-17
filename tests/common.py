"""
Pytest fixtures. Due to the heterogeneity between tests, most fixtures are factories.
"""
import itertools
import os

N_LOCAL_DEVICE = 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_LOCAL_DEVICE}"

import dataclasses
import jax
import jax.numpy as jnp
import pytest
import seqio
import tensorflow as tf

from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.attn2 import get_attn_cls
from transformer_vq.nn.model import Transformer

jax.config.update("jax_enable_x64", True)


ATTN_TYPES = ["vq", "full", "vq_old", "full_old"]  # the _old ones use a vanilla scan
ATTN_TYPES_OLD_UNWRAPPED = ["vq_old_unwrapped", "full_old_unwrapped"]
HEAD_TYPES = ["mha", "mqa", "shga"]
REDUCTION_TYPES = ["serial", "matmul", "assoc_scan", "sum"]  # sum is unstable
WIDENINGS = [7]
DTYPES = [jnp.float32]
TOLERANCES = dict(atol=1e-5, rtol=1e-4)


def get_list(given, default):
    if given == "all":
        return default
    if given is None:
        return [None]
    if isinstance(given, str):
        return [given]
    if isinstance(given, list):
        return given
    raise NotImplementedError("Unexpected value for given settings.")


def gen_type_tuples(attn_type="all", head_type="all", reduction_type="all"):
    attn_types = get_list(given=attn_type, default=ATTN_TYPES)
    head_types = get_list(given=head_type, default=HEAD_TYPES)
    reduction_types = get_list(given=reduction_type, default=REDUCTION_TYPES)
    prod = itertools.product(attn_types, head_types, reduction_types)
    out = []
    for a, h, r in prod:
        if a != "vq":
            r = None
        out.append((a, h, r))
    return out


def gen_len_tuples(t=12):
    tuples = []
    for ell in range(1, t):
        if t % ell == 0:
            tuples.append((t, ell))
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
        sequence_len,
        block_len,
        widening,
        dtypes,
        is_train,
        n_vocab=10,
        c_gamma=0.99,
        no_emb=False,
        global_batch_size=None,  # set to none if not used
        attn_type="vq",
        head_type="shga",
        reduction_type="matmul",
    ):
        return TransformerConfig.create(
            n_device=N_LOCAL_DEVICE,
            param_dtype=dtypes,
            dtype=dtypes,
            global_batch_size=global_batch_size,
            sequence_len=sequence_len,
            block_len=block_len,
            attn_type=attn_type,
            head_type=head_type,
            reduction_type=reduction_type,
            d_model=4 * widening,
            d_k=2 * widening,
            n_code=8,
            n_layer=2,
            n_vocab=n_vocab,
            pe_abs=True,
            pe_lam=10_000,
            p_dropsin=0.0,
            p_dropres=0.0,
            p_droplyr=0.0,
            c_beta=0.02,
            c_gamma=c_gamma,
            e_tie=True,
            e_preln=True,
            e_scale=0.005,
            is_train=is_train,
            no_emb=no_emb,
        )

    return _transformer_config


@pytest.fixture
def attn_fixture(rng_fixture):
    def _attn(config: TransformerConfig):
        config = dict(dataclasses.asdict(config).items())  # make a copy
        config.update({"is_train": True})
        config = TransformerConfig.create(**config)
        attn_cls = get_attn_cls(config.attn_type)
        inputs = jax.random.normal(
            key=rng_fixture(4),
            shape=[1, config.sequence_len, config.d_model],
            dtype=config.dtype,
        )
        if config.attn_type.endswith("old_unwrapped"):
            state_kwargs = dict(state=attn_cls.initial_state(config, batch_size=1))
        else:
            state_kwargs = dict()
            inputs = attn_cls.pre_reshape(inputs, config)
        params = attn_cls(config).init(
            dict(
                params=rng_fixture(1),
                ephemeral=rng_fixture(2),
                timeless=rng_fixture(3),
            ),
            **state_kwargs,
            x=inputs,
        )["params"]
        return attn_cls, params

    return _attn


@pytest.fixture
def transformer_fixture(rng_fixture):
    def _transformer(config):
        config = dict(dataclasses.asdict(config).items())  # make a copy
        config.update({"is_train": True})
        config = TransformerConfig.create(**config)
        params = Transformer(config).init(
            dict(
                params=rng_fixture(1),
                ephemeral=rng_fixture(2),
                timeless=rng_fixture(3),
            ),
            inputs=jax.random.randint(
                key=rng_fixture(0),
                minval=0,
                maxval=config.n_vocab,
                shape=[1, config.sequence_len],
            ),
        )["params"]
        return Transformer, params

    return _transformer


@pytest.fixture
def vocab_fixture():
    return seqio.ByteVocabulary()


@pytest.fixture
def img_fixture():
    dir1 = os.path.dirname(os.path.realpath(__file__))
    fname1 = "ref_img.png"
    with tf.io.gfile.GFile(tf.io.gfile.join(dir1, fname1), "rb") as f1:
        data1 = f1.read()
    return tf.io.decode_png(data1, channels=3)
