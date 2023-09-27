import os
import tempfile
import time
from typing import List
from typing import Optional

import jax
import sentencepiece as spm
import tensorflow as tf
import tensorflow_datasets as tfds


def dump_chars_to_tempfile(ds, maxchars):
    char_count = 0
    with tempfile.NamedTemporaryFile(delete=False, prefix="/tmp/ds_chars") as outfp:
        for document_chars in ds:
            if (maxchars is not None) and (char_count >= maxchars):
                break
            outfp.write(document_chars + b"\n")
            char_count += len(document_chars)
        return outfp.name, char_count


def maybe_train_sentencepiece_model(
    tfds_data_dir: Optional[str],
    tfds_id: str,
    tfds_col: str,
    spm_vocab_path: str,
    spm_vocab_size: int,
    spm_uds: List[str],
    maxchars: Optional[int] = int(1e9),
):
    if spm_vocab_path.startswith("gs://"):
        abs_spm_vocab_path = spm_vocab_path
    else:
        abs_spm_vocab_path = os.path.abspath(os.path.expanduser(spm_vocab_path))
        os.makedirs(os.path.dirname(abs_spm_vocab_path), exist_ok=True)
    if tf.io.gfile.exists(abs_spm_vocab_path):
        return abs_spm_vocab_path
    if jax.process_index() == 0:
        chardump_ds = (
            tfds.load(tfds_id, split="train", data_dir=tfds_data_dir, try_gcs=True)
            .map(lambda r: r[tfds_col])
            .as_numpy_iterator()
        )
        fname, _ = dump_chars_to_tempfile(ds=chardump_ds, maxchars=maxchars)
        with tempfile.NamedTemporaryFile(delete=False, prefix="/tmp/sp_tmp") as temp_fp:
            # for consistency with seqio.ByteVocabulary, we use 0=PAD, 1=EOS, 2=UNK
            spm.SentencePieceTrainer.Train(
                input=fname,
                vocab_size=spm_vocab_size,
                character_coverage=1.0,
                model_prefix=temp_fp.name,
                model_type="bpe",
                user_defined_symbols=spm_uds,
                pad_id=0,
                eos_id=1,
                unk_id=2,
                bos_id=-1,
            )
            copy_rename_path = abs_spm_vocab_path + ".rntmp"
            print(temp_fp.name + ".model")
            tf.io.gfile.copy(temp_fp.name + ".model", copy_rename_path, overwrite=True)
            tf.io.gfile.rename(copy_rename_path, abs_spm_vocab_path, overwrite=True)
    else:
        while not tf.io.gfile.exists(abs_spm_vocab_path):
            time.sleep(1)
        time.sleep(1)
    return abs_spm_vocab_path
