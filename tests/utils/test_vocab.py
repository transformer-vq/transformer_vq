import tensorflow as tf

from transformer_vq.utils.vocab import dump_chars_to_tempfile


def test_dump_chars_to_tempfile():
    ds = (
        tf.data.Dataset.from_tensors(tf.constant(["abc", "def", "ghi", "jkl"]))
        .unbatch()
        .as_numpy_iterator()
    )
    fp, count = dump_chars_to_tempfile(ds, maxchars=10)
    actual_chars = []
    with open(fp, "rb") as f:
        for line in f:
            actual_chars.append(line)
    actual_chars = b"".join(actual_chars)
    assert actual_chars == b"abc\ndef\nghi\njkl\n"
