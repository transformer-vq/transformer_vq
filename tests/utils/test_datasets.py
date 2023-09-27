import numpy as np
import pytest
import seqio
import tensorflow as tf

# noreorder
from transformer_vq.utils.datasets import Dataset
from transformer_vq.utils.datasets import image_flatten
from transformer_vq.utils.datasets import image_offset
from tests.utils.fixtures import img


def test_byte_vocab_specials():
    vocab = seqio.ByteVocabulary()
    assert vocab.pad_id == 0
    assert vocab.eos_id == 1
    assert vocab.unk_id == 2
    with pytest.raises(NotImplementedError):
        print(f"byte-level bos_id: {vocab.bos_id}")


def test_img_pipeline(img):
    # encode
    x = {"targets": tf.constant(img, dtype=tf.uint8)}
    x = image_flatten(x)
    x = image_offset(x)["targets"]
    # decode
    y = Dataset.decode_image(x, tuple(img.shape))
    # check
    np.testing.assert_allclose(actual=y, desired=img)
