from typing import Iterator
from typing import Tuple

import jax
import numpy as np
import seqio
import tensorflow as tf


def pack_examples(ds, sequence_len, vocab, append_eos, is_train):
    # pack documents into sequences of specified length. for image datasets,
    # we will use seq_len = prod(img_shape) and append_eos=False to make this a no-op.
    def func(r):
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.eos_id)
        return tf.pad(r, [[0, 1]], **pad_kwargs)

    if append_eos:
        ds = ds.map(func)
    ds = ds.unbatch()
    # to avoid loss masking on codebook loss, we will drop any remainder during training
    # after wrapping/packing sequences into as many sequence_len sequences as we can
    ds = ds.batch(batch_size=sequence_len, drop_remainder=is_train)
    return ds


def pad_examples(ds, sequence_len, vocab):
    # pad examples to length sequence_len.
    # only not a no-op for a possible 'remainder' sequence after packing.
    # always a no-op with images, since packing is a no-op w/ them and has no remainder.
    def func(r):
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.pad_id)
        return tf.pad(r, [[0, sequence_len - tf.shape(r)[0]]], **pad_kwargs)

    return ds.map(func)


def examples_to_features(ds, sequence_len, vocab):
    def func(r):
        # use eos_id to prompt the sequence; we assume vocab has no bos_id.
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.eos_id)
        inputs = tf.pad(r[:-1], [[1, 0]], **pad_kwargs)
        targets = r
        loss_mask = targets != vocab.pad_id
        return dict(
            inputs=tf.ensure_shape(inputs, [sequence_len]),
            targets=tf.ensure_shape(targets, [sequence_len]),
            loss_mask=tf.ensure_shape(loss_mask, [sequence_len]),
        )

    return ds.map(func)


def pad_batches(ds, batch_size, sequence_len, vocab):
    # pad batches with too few examples
    def func(r):
        pad_spec = [[0, batch_size - tf.shape(r["targets"])[0]], [0, 0]]
        # pad below of input and target with vocab.pad_id
        pad_kwargs = dict(mode="CONSTANT", constant_values=vocab.pad_id)
        r["inputs"] = tf.pad(r["inputs"], pad_spec, **pad_kwargs)
        r["targets"] = tf.pad(r["targets"], pad_spec, **pad_kwargs)
        # pad below of loss_mask, doc_ids with zeros.
        pad_kwargs = dict(mode="CONSTANT", constant_values=False)
        r["loss_mask"] = tf.pad(r["loss_mask"], pad_spec, **pad_kwargs)
        # check shapes
        output_shape = [batch_size, sequence_len]
        return dict(
            inputs=tf.ensure_shape(r["inputs"], output_shape),
            targets=tf.ensure_shape(r["targets"], output_shape),
            loss_mask=tf.ensure_shape(r["loss_mask"], output_shape),
        )

    return ds.map(func)


def get_batches(
    ds: tf.data.Dataset,
    batch_size: int,
    sequence_len: int,
    is_train: bool,
    vocab: seqio.Vocabulary,
    append_eos: bool,
    shuffle_seed_root: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    # extract batches from dataset.
    # each batch is a tuple containing four tensors of shape [batch_size, sequence_len].
    # the tensors are the inputs, targets, document ids, and loss mask.
    options = tf.data.Options()
    options.autotune.enabled = True
    common = dict(sequence_len=sequence_len, vocab=vocab)
    ds = pack_examples(ds, append_eos=append_eos, is_train=is_train, **common)
    ds = pad_examples(ds, **common)
    ds = examples_to_features(ds, **common)
    if is_train:
        # loop infinitely, yield batches of size batch_size
        ds = ds.repeat()
        shuffle_seed = int(shuffle_seed_root + (10**9) * jax.process_index())
        ds = ds.shuffle(buffer_size=10_000, seed=shuffle_seed)
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    else:
        # yield all batches, padding smaller ones to batch_size for full eval coverage
        ds = ds.batch(batch_size=batch_size, drop_remainder=False)
        ds = pad_batches(ds, batch_size=batch_size, **common)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds.as_numpy_iterator()
