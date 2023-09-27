import os

import pytest
import seqio
import tensorflow as tf


@pytest.fixture
def vocab():
    return seqio.ByteVocabulary()


@pytest.fixture
def img():
    dir1 = os.path.dirname(os.path.realpath(__file__))
    fname1 = "ref_img.png"
    with tf.io.gfile.GFile(tf.io.gfile.join(dir1, fname1), "rb") as f1:
        data1 = f1.read()
    return tf.io.decode_png(data1, channels=3)
