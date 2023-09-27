import os
import tempfile

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tensorflow as tf
from flax.training.train_state import TrainState

from transformer_vq.utils.io import check_not_none
from transformer_vq.utils.io import load_checkpoint
from transformer_vq.utils.io import save_checkpoint
from transformer_vq.utils.io import save_pixels

# noreorder
from tests.common import rng_fixture


STEPS = 7357
N_VOCAB = 123
BATCH_SIZE = 1
SEQUENCE_LEN = 100


def test_check_not_none():
    check_not_none("something")
    with pytest.raises(ValueError):
        check_not_none(None)


class Model(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x = nn.Embed(N_VOCAB, 1000)(inputs)
        x = nn.relu(x)
        x = nn.Dense(1000)(x)
        x = nn.relu(x)
        x = nn.Dense(N_VOCAB)(x)
        return x


@pytest.fixture
def train_state():
    def _train_state(init_rng):
        model = Model()
        sk1, sk2 = jax.random.split(init_rng)
        params = model.init(
            {"params": sk1},
            inputs=jnp.ones(dtype=jnp.int32, shape=[BATCH_SIZE, SEQUENCE_LEN]),
        )["params"].unfreeze()
        tx = optax.sgd(learning_rate=0.01)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        state = state.replace(step=STEPS)
        return state

    return _train_state


def test_save_checkpoint(rng_fixture, train_state, tmp_path):
    train_state = train_state(rng_fixture(0))
    save_checkpoint(
        save_dir=str(tmp_path),
        target=train_state,
        step=STEPS,
        prefix="checkpoint",
        keep=5,
    )
    assert os.path.exists(os.path.join(tmp_path, f"checkpoint_{STEPS}"))


def test_load_checkpoint(rng_fixture, train_state, tmp_path):
    state = train_state(rng_fixture(0))
    state_bytes = flax.serialization.to_bytes(state)
    save_checkpoint(
        save_dir=str(tmp_path),
        target=state,
        step=STEPS,
        prefix="checkpoint",
        keep=5,
    )
    assert os.path.exists(os.path.join(tmp_path, f"checkpoint_{STEPS}"))
    state_loaded = load_checkpoint(
        load_dir=str(tmp_path),
        train_state=state,
        prefix="checkpoint",
    )
    state_bytes_loaded = flax.serialization.to_bytes(state_loaded)
    assert state_bytes == state_bytes_loaded


def test_save_pixels():
    dir1 = os.path.dirname(os.path.realpath(__file__))
    fname1 = "ref_img.png"  # example image taken from reddit's /r/programmerhumor
    with tf.io.gfile.GFile(tf.io.gfile.join(dir1, fname1), "rb") as f1:
        data1 = f1.read()
    data1 = tf.io.decode_png(data1)

    tempdir = tempfile.TemporaryDirectory()
    dir2 = tempdir.name
    fname2 = "saved_pixels.png"
    save_pixels(data1, dir2, fname2)

    with tf.io.gfile.GFile(tf.io.gfile.join(dir2, fname2), "rb") as f2:
        data2 = f2.read()
    data2 = tf.io.decode_png(data2)
    np.testing.assert_allclose(data2, data1)
    tempdir.cleanup()
