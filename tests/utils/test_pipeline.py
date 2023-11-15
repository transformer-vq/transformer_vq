import jax
import numpy as np
import pytest
import tensorflow as tf

# noreorder
from transformer_vq.utils.pipeline import pack_examples
from transformer_vq.utils.pipeline import pad_examples
from transformer_vq.utils.pipeline import examples_to_features
from transformer_vq.utils.pipeline import pad_batches
from tests.common import vocab_fixture


@pytest.mark.parametrize("append_eos", [True, False])
def test_pack_examples(vocab_fixture, append_eos):
    ds = tf.data.Dataset.from_tensors(tf.range(7))
    ds = pack_examples(
        ds, sequence_len=5, vocab=vocab_fixture, append_eos=append_eos, is_train=False
    )
    actual = list(ds.as_numpy_iterator())
    actual_row0 = actual[0]
    actual_row1 = actual[1]
    expected_row0 = tf.constant([0, 1, 2, 3, 4])
    expected_row1 = tf.constant([5, 6])
    if append_eos:
        expected_row1 = tf.pad(
            expected_row1, [[0, 1]], constant_values=vocab_fixture.eos_id
        )
    np.testing.assert_allclose(actual=actual_row0, desired=expected_row0)
    np.testing.assert_allclose(actual=actual_row1, desired=expected_row1)


@pytest.mark.parametrize("append_eos", [True, False])
def test_pad_examples(vocab_fixture, append_eos):
    ds = tf.data.Dataset.from_tensors(tf.range(7))
    ds = pack_examples(
        ds, sequence_len=5, vocab=vocab_fixture, append_eos=append_eos, is_train=False
    )
    ds = pad_examples(ds, sequence_len=5, vocab=vocab_fixture)
    actual = list(ds.as_numpy_iterator())
    actual_row0 = actual[0]
    actual_row1 = actual[1]
    expected_row0 = tf.constant([0, 1, 2, 3, 4])
    expected_row1 = tf.constant([5, 6])
    pad_id = vocab_fixture.pad_id
    eos_id = vocab_fixture.eos_id
    if append_eos:
        expected_row1 = tf.pad(expected_row1, [[0, 1]], constant_values=eos_id)
        expected_row1 = tf.pad(expected_row1, [[0, 2]], constant_values=pad_id)
    else:
        expected_row1 = tf.pad(expected_row1, [[0, 3]], constant_values=pad_id)
    np.testing.assert_allclose(actual=actual_row0, desired=expected_row0)
    np.testing.assert_allclose(actual=actual_row1, desired=expected_row1)


def test_examples_to_features(vocab_fixture):
    pad_id = vocab_fixture.pad_id
    eos_id = vocab_fixture.eos_id
    ds = tf.data.Dataset.from_tensors(tf.constant([10, 20, 30, pad_id, pad_id]))
    ds = examples_to_features(ds=ds, sequence_len=5, vocab=vocab_fixture)
    actual = list(ds.as_numpy_iterator())[0]
    np.testing.assert_allclose(
        actual=actual["inputs"],
        desired=tf.constant([eos_id, 10, 20, 30, pad_id]),
    )
    np.testing.assert_allclose(
        actual=actual["targets"],
        desired=tf.constant([10, 20, 30, pad_id, pad_id]),
    )
    np.testing.assert_allclose(
        actual=actual["loss_mask"],
        desired=tf.constant([1, 1, 1, 0, 0]),
    )


def test_pad_batches(vocab_fixture):
    pad_id = vocab_fixture.pad_id
    expected = dict(
        inputs=tf.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, pad_id], [pad_id] * 5]),
        targets=tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, pad_id, pad_id], [pad_id] * 5]),
        loss_mask=tf.constant(
            [[True] * 5, [True, True, True, False, False], [False] * 5]
        ),
    )
    unpadded = jax.tree_util.tree_map(lambda x: x[0:2, :], expected)
    ds = tf.data.Dataset.from_tensors(unpadded)
    ds = pad_batches(ds=ds, batch_size=3, sequence_len=5, vocab=vocab_fixture)
    actual = list(ds.as_numpy_iterator())[0]
    for key in actual:
        print(key)
        np.testing.assert_allclose(actual=actual[key], desired=expected[key])
