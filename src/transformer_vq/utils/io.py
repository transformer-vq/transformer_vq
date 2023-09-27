import tensorflow as tf
from flax.training import checkpoints


def check_not_none(s):
    if s is None:
        raise ValueError("argument cannot be None.")


def save_checkpoint(target, save_dir, prefix, step, keep):
    check_not_none(save_dir)
    checkpoints.save_checkpoint_multiprocess(
        save_dir,
        target=target,
        step=step,
        prefix=f"{prefix}_",
        keep=keep,
        overwrite=False,
        keep_every_n_steps=None,
        async_manager=None,
        orbax_checkpointer=None,
    )


def load_checkpoint(train_state, load_dir, prefix):
    check_not_none(load_dir)
    train_state = checkpoints.restore_checkpoint(
        ckpt_dir=load_dir,
        target=train_state,
        prefix=prefix,
        step=None,
    )
    return train_state


def save_text(target, dirname, fname, mode="w"):
    fp = tf.io.gfile.join(dirname, fname)
    with tf.io.gfile.GFile(fp, mode=mode) as f:
        f.write(target)
        f.flush()


def save_pixels(target, dirname, fname):
    target = tf.io.encode_png(target).numpy()
    save_text(target=target, dirname=dirname, fname=fname, mode="wb")
