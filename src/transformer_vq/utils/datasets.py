"""
Dataset definitions.
"""
import abc
import functools
from typing import Dict
from typing import Iterable
from typing import Tuple

import jax
import numpy as np
import seqio
import tensorflow as tf

from transformer_vq.utils.pipeline import get_batches
from transformer_vq.utils.vocab import maybe_train_sentencepiece_model


# All datasets use vocabularies where 0=pad, 1=eos, 2=unk.
NUM_SPECIAL = 3


def image_flatten(r):
    """for flattening images"""
    return {"targets": tf.cast(tf.reshape(r["targets"], [-1]), tf.int32)}


def image_offset(r):
    """
    for use with flattened images.
    this is equivalent to converting the bytes to a string.
    then encoding them as a tensor using seqio.ByteVocab(),
    which introduces special tokens 0, 1, 2 and offsets all valid bytes by 3.
    """
    return {"targets": r["targets"] + NUM_SPECIAL}


def add_to_seqio(task_name, tfds_data_dir, tfds_id, tfds_col, vocab, modality):
    preprocessors = list()
    preprocessors.append(
        functools.partial(seqio.preprocessors.rekey, key_map={"targets": tfds_col})
    )
    if modality == "image":
        preprocessors.append(seqio.utils.map_over_dataset(image_flatten))
        preprocessors.append(seqio.utils.map_over_dataset(image_offset))
    if modality == "text":
        preprocessors.append(seqio.preprocessors.tokenize)
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TfdsDataSource(tfds_id, tfds_data_dir=tfds_data_dir),
        preprocessors=preprocessors,
        output_features={"targets": seqio.Feature(vocabulary=vocab)},
    )


class Dataset(metaclass=abc.ABCMeta):
    task_name: str
    modality: str
    registry: Dict = dict()

    def __init__(
        self,
        tfds_data_dir,
        tfds_id,
        tfds_col,
        spm_vocab_path,
        spm_vocab_size,
        spm_uds,
        img_shape=None,
    ):
        self.tfds_data_dir = tfds_data_dir
        self.tfds_id = tfds_id
        self.tfds_col = tfds_col
        self.spm_vocab_path = spm_vocab_path
        self.spm_vocab_size = spm_vocab_size
        self.spm_uds = spm_uds
        self.img_shape = img_shape
        self.vocab = self._get_vocab()

    def __init_subclass__(cls, **kwargs):
        """adds subclasses to registry when they are defined -- see pep 487"""
        super().__init_subclass__(**kwargs)
        cls.registry[cls.task_name] = cls

    def _get_vocab(self):
        if self.spm_vocab_size is None:
            return seqio.ByteVocabulary()
        else:
            vocab_path = maybe_train_sentencepiece_model(
                tfds_data_dir=self.tfds_data_dir,
                tfds_id=self.tfds_id,
                tfds_col=self.tfds_col,
                spm_vocab_path=self.spm_vocab_path,
                spm_vocab_size=self.spm_vocab_size,
                spm_uds=self.spm_uds,
            )
            return seqio.SentencePieceVocabulary(vocab_path)

    @staticmethod
    def _get_shard_info():
        host_id = jax.process_index()
        n_host = jax.process_count()
        return seqio.ShardInfo(index=host_id, num_shards=n_host)

    def _get_tokenized(self, split_name, shard_by_host):
        if self.task_name not in seqio.TaskRegistry.names():
            add_to_seqio(
                task_name=self.task_name,
                tfds_data_dir=self.tfds_data_dir,
                tfds_id=self.tfds_id,
                tfds_col=self.tfds_col,
                vocab=self.vocab,
                modality=self.modality,
            )
        task = seqio.TaskRegistry.get(self.task_name)
        shard_info = Dataset._get_shard_info() if shard_by_host else None
        ds = task.get_dataset(
            sequence_length=None,
            split=split_name,
            shuffle=False,
            shard_info=shard_info,
        )
        return ds.map(lambda x: tf.cast(x["targets"], tf.int32))

    @property
    def vocab_size(self):
        return self.vocab.vocab_size

    @staticmethod
    def decode_text(ints: Iterable[int], vocab: seqio.Vocabulary):
        return vocab.decode(ints)

    @staticmethod
    def decode_image(ints: Iterable[int], image_shape: Tuple[int, int, int]):
        assert isinstance(image_shape, tuple) and len(image_shape) == 3
        ints = [i - NUM_SPECIAL for i in ints if i >= NUM_SPECIAL]
        return np.reshape(np.array(ints, dtype=np.uint8), image_shape)

    def decode(self, ints):
        if self.modality == "text":
            return Dataset.decode_text(ints, self.vocab)
        if self.modality == "image":
            return Dataset.decode_image(ints, self.img_shape)
        raise NotImplementedError

    @abc.abstractmethod
    def get_iter(self, split_name, batch_size, sequence_len, shuffle_seed_root):
        raise NotImplementedError


class Enwik8(Dataset):
    task_name = "enwik8"
    modality = "text"

    def __init__(self, vocab_path, data_dir):
        super().__init__(
            tfds_data_dir=data_dir,
            tfds_id="huggingface:enwik8/enwik8-raw:1.1.0",
            tfds_col="text",
            spm_vocab_path=None,  # bytes
            spm_vocab_size=None,  # bytes
            spm_uds=[],
        )

    def get_iter(self, split_name, batch_size, sequence_len, shuffle_seed_root):
        # split manually, since huggingface dataset gives all 100m enwik8 bytes as train
        ds = self._get_tokenized("train", shard_by_host=False)  # never shard; small
        token_ints = list(ds.take(1).as_numpy_iterator())[0]
        split_indices = [90_000_000, 95_000_000]
        split_names = ["train", "validation", "test"]
        splits = np.split(token_ints, split_indices)
        ds = tf.data.Dataset.from_tensors(dict(zip(split_names, splits))[split_name])
        return get_batches(
            ds=ds.map(lambda r: tf.cast(r, tf.int32)),
            batch_size=batch_size,
            sequence_len=sequence_len,
            is_train=split_name == "train",
            vocab=self.vocab,
            append_eos=False,
            shuffle_seed_root=shuffle_seed_root,
        )


class PG19(Dataset):
    task_name = "pg19"
    modality = "text"

    def __init__(self, vocab_path, data_dir):
        super().__init__(
            tfds_data_dir=data_dir,
            tfds_id="pg19:0.1.1",
            tfds_col="book_text",
            spm_vocab_path=vocab_path,
            spm_vocab_size=32_000,
            spm_uds=[],
        )

    def get_iter(self, split_name, batch_size, sequence_len, shuffle_seed_root):
        return get_batches(
            ds=self._get_tokenized(split_name, shard_by_host=split_name == "train"),
            batch_size=batch_size,
            sequence_len=sequence_len,
            is_train=split_name == "train",
            vocab=self.vocab,
            append_eos=True,
            shuffle_seed_root=shuffle_seed_root,
        )


class Imagenet64(Dataset):
    task_name = "imagenet64"
    modality = "image"

    def __init__(self, vocab_path, data_dir):
        super().__init__(
            tfds_data_dir=data_dir,
            tfds_id="imagenet_resized/64x64:0.1.0",
            tfds_col="image",
            spm_vocab_path=None,  # bytes
            spm_vocab_size=None,  # bytes
            spm_uds=[],
            img_shape=(64, 64, 3),
        )

    def get_iter(self, split_name, batch_size, sequence_len, shuffle_seed_root):
        if split_name == "train":
            # use first 1.2m examples of official training set for training
            ds = self._get_tokenized("train", shard_by_host=True)
            ds = ds.take(1_200_000)
            return get_batches(
                ds=ds,
                batch_size=batch_size,
                sequence_len=sequence_len,
                is_train=True,
                vocab=self.vocab,
                append_eos=False,
                shuffle_seed_root=shuffle_seed_root,
            )
        if split_name == "validation":
            # use remaining examples in training split for validation
            ds = self._get_tokenized("train", shard_by_host=False)
            ds = ds.skip(1_200_000)
            return get_batches(
                ds=ds,
                batch_size=batch_size,
                sequence_len=sequence_len,
                is_train=False,
                vocab=self.vocab,
                append_eos=False,
                shuffle_seed_root=shuffle_seed_root,
            )
        if split_name == "test":
            # use official validation set to benchmark; imagenet has no public test set
            ds = self._get_tokenized("validation", shard_by_host=False)
            return get_batches(
                ds=ds,
                batch_size=batch_size,
                sequence_len=sequence_len,
                is_train=False,
                vocab=self.vocab,
                append_eos=False,
                shuffle_seed_root=shuffle_seed_root,
            )
        raise NotImplementedError
