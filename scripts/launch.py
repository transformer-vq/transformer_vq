import functools
import os.path as osp  # etils.epath currently gives errors with local paths
import sys
import time
from typing import Any
from typing import Dict
from typing import Optional

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import wandb
from absl import app
from absl import flags
from absl import logging
from etils import epath
from flax import jax_utils
from flax.training import common_utils
from flax.training import orbax_utils
from flax.training import train_state as train_utils
from ml_collections import config_flags

from transformer_vq.nn.model import Transformer
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.utils.datasets import Dataset
from transformer_vq.utils.tree import flattened_traversal

MODES = ["train_vocab", "train", "validation", "test", "flop_count"]

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Configuration file", lock_config=False)
flags.DEFINE_boolean("multihost", None, "Multihost?")
flags.DEFINE_enum("mode", None, MODES, "Mode.")
flags.DEFINE_string("workdir", None, "Directory for experiment data.")
flags.DEFINE_boolean("wb_enabled", False, "Log to W&B?")
flags.DEFINE_string("wb_run", None, "W&B run id, for resuming with continuity.")
flags.DEFINE_string("model_name", None, "Optional model name. Generated if not given.")
flags.DEFINE_string("gpu_coord_addr", None, "For non-slurm/openmpi GPU clusters.")
flags.DEFINE_string("gpu_n_process", None, "For non-slurm/openmpi GPU clusters.")
flags.DEFINE_string("gpu_process_id", None, "For non-slurm/openmpi GPU clusters.")
flags.mark_flags_as_required(["config", "multihost", "workdir", "mode"])


def get_model_name() -> str:
    if FLAGS.model_name is not None:
        return FLAGS.model_name
    a = FLAGS.config.attn_type
    h = FLAGS.config.head_type
    return f"transformer_vq_{FLAGS.config.dataset}_{a}_{h}"


def get_vocab_path() -> str:
    fname = f"{FLAGS.config.dataset}_sentencepiece.model"
    return osp.join(FLAGS.workdir, "vocabs", fname)


def get_data_dir() -> str:
    return osp.join(FLAGS.workdir, "datasets")


def get_checkpoint_manager() -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        directory=epath.Path(FLAGS.workdir) / "checkpoints" / get_model_name(),
        checkpointers=ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=5,
            save_interval_steps=1,
            step_prefix="state",
        ),
    )


def do_restore(
    mgr: ocp.CheckpointManager,
    target: train_utils.TrainState,
) -> train_utils.TrainState:
    return mgr.restore(
        step=mgr.latest_step(),
        items=target,
        restore_kwargs=dict(
            restore_args=orbax_utils.restore_args_from_target(target, mesh=None),
        ),
    )


def do_save(
    mgr: ocp.CheckpointManager,
    target: train_utils.TrainState,
    step: int,
) -> None:
    mgr.save(
        step=step,
        items=target,
        save_kwargs=dict(
            save_args=orbax_utils.save_args_from_target(target),
        ),
    )


def get_root_rngs():
    rng_root = jax.random.PRNGKey(FLAGS.config.prng_seed)
    rng_sync, rng_unsync = jax.random.split(rng_root)
    rng_unsync = jax.random.fold_in(rng_unsync, jax.process_index())
    return rng_sync, rng_unsync


def get_transformer_config_dict(is_train) -> Dict[str, Any]:
    return {
        **vars(FLAGS.config)["_fields"],
        "is_train": is_train,
        "n_device": jax.device_count(),
    }


def get_transformer_config(is_train) -> TransformerConfig:
    return TransformerConfig.create(**get_transformer_config_dict(is_train))


def get_optimizer() -> optax.GradientTransformation:
    if FLAGS.config.optimizer == "adamw":
        return optax.adamw(
            learning_rate=FLAGS.config.lr_max,
            b1=0.9,
            b2=0.98,
            eps=10**-9,
            mu_dtype=FLAGS.config.param_dtype,  # Rae et al., 2021
            weight_decay=0.0,  # optimizers in optax scale wd by lr, so diy
        )
    if FLAGS.config.optimizer == "lion":
        return optax.lion(
            learning_rate=FLAGS.config.lr_max,
            b1=0.95,
            b2=0.98,
            mu_dtype=FLAGS.config.dtype,  # Chen et al., 2023
            weight_decay=0.0,  # optimizers in optax scale wd by lr, so diy
        )
    if FLAGS.config.optimizer == "adafactor":
        return optax.adafactor(
            learning_rate=FLAGS.config.lr_max,
            multiply_by_parameter_scale=True,
            clipping_threshold=1.0,  # must be >= 1.0 per optax docs.
            weight_decay_rate=0.0,  # optimizers in optax scale wd by lr, so diy
        )


def get_schedule_fn() -> optax.Schedule:
    max_steps = FLAGS.config.n_max_step
    warmup_steps = FLAGS.config.n_warmup_step
    decay_steps = max_steps - warmup_steps  # exclude n_extra
    warmup = optax.linear_schedule(0.0, 1.0, transition_steps=warmup_steps)
    if FLAGS.config.lr_schedule == "cosine":
        main_schedule = optax.cosine_decay_schedule(
            init_value=1.0,
            decay_steps=decay_steps,
            alpha=0.1,  # final schedule value
        )
    elif FLAGS.config.lr_schedule == "inv_sqrt":
        main_schedule = optax.polynomial_schedule(
            init_value=1.0,
            end_value=0.1,
            power=-0.5,
            transition_steps=decay_steps,
        )
    elif FLAGS.config.lr_schedule == "linear":
        main_schedule = optax.linear_schedule(
            init_value=1.0,
            end_value=0.1,
            transition_steps=decay_steps,
        )
    else:
        raise NotImplementedError("schedule name not supported")
    return optax.join_schedules(
        schedules=[warmup, main_schedule], boundaries=[FLAGS.config.n_warmup_step]
    )


def get_param_label_fn():
    return flattened_traversal(
        lambda path, _: "main" if not path[-1].startswith("c_") else "codebook"
    )


def get_tx() -> optax.GradientTransformation:
    maybe_gradclip = []
    if FLAGS.config.grad_clip is not None:
        maybe_gradclip.append(optax.clip_by_global_norm(FLAGS.config.grad_clip))
    optimizer = get_optimizer()
    schedule = optax.scale_by_schedule(get_schedule_fn())
    maybe_decay = []
    if FLAGS.config.wd_lam > 0.0:
        wd = optax.add_decayed_weights(
            weight_decay=-FLAGS.config.wd_lam,
            mask=lambda p: jax.tree_util.tree_map(lambda x: jnp.ndim(x) != 1, p),
        )
        maybe_decay.append(wd)
    tx = optax.multi_transform(
        {
            "main": optax.chain(*maybe_gradclip, optimizer, schedule, *maybe_decay),
            "codebook": optax.sgd(learning_rate=1.0),
        },
        param_labels=get_param_label_fn(),
    )
    return tx


def get_train_state(rng_init):
    # make fast init config
    canary = FLAGS.config.sequence_len
    assert canary != 8
    config = get_transformer_config_dict(is_train=True)
    config.update({"sequence_len": 24, "block_len": 8, "mem_len": 8})  # need > 2 blocks
    config = TransformerConfig.create(**config)
    assert FLAGS.config.sequence_len == canary
    # do init
    rng_param, rng_ephemeral, rng_timeless = jax.random.split(rng_init, 3)
    params = Transformer(config).init(
        dict(params=rng_param, ephemeral=rng_ephemeral, timeless=rng_timeless),
        inputs=jnp.zeros([1, config.sequence_len], dtype=jnp.int32),
    )["params"]
    tx = get_tx()
    # log the param count
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"param count: {param_count}")
    # return train state
    return train_utils.TrainState.create(apply_fn=None, params=params.unfreeze(), tx=tx)


def get_dataset():
    dataset_cls = Dataset.registry.get(FLAGS.config.dataset)
    dataset = dataset_cls(vocab_path=get_vocab_path(), data_dir=get_data_dir())
    if FLAGS.mode == "train_vocab":
        sys.exit(0)
    setattr(FLAGS.config, "n_vocab", dataset.vocab_size)
    return dataset


def loss_fn(params, config, batch, rng):
    rngs = dict()
    if rng is not None:
        rng_ephemeral, rng_timeless = jax.random.split(rng, 2)
        rngs = dict(ephemeral=rng_ephemeral, timeless=rng_timeless)
    outputs = Transformer(config).apply({"params": params}, batch["inputs"], rngs=rngs)
    l_ce_terms_unmasked = optax.softmax_cross_entropy_with_integer_labels(
        logits=outputs["logits"], labels=batch["targets"]
    )
    l_ce_term_avg = jnp.mean(batch["loss_mask"] * l_ce_terms_unmasked)
    l_ce_mask_avg = jnp.mean(batch["loss_mask"])
    l_commit_avg = outputs["l_commit"]
    l_codebook_avg = outputs["l_codebook"]
    l_total_avg = l_ce_term_avg + config.c_beta * l_commit_avg + l_codebook_avg
    metrics = dict(
        l_ce_term_avg=l_ce_term_avg,
        l_ce_mask_avg=l_ce_mask_avg,
        l_commit_avg=l_commit_avg,
        l_codebook_avg=l_codebook_avg,
    )
    return l_total_avg, metrics


@functools.partial(jax.pmap, donate_argnums=(0,), axis_name="devices")
def train_step(train_state, batch, rng):
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params,
        config=get_transformer_config(is_train=True),
        batch=batch,
        rng=rng,
    )
    metrics, grads = jax.lax.pmean([metrics, grads], axis_name="devices")
    train_state = train_state.apply_gradients(grads=grads)
    # Estimate ce loss for global batch: sum of unmasked ce terms / sum of mask values.
    # Equivalent to:
    metrics["l_ce_avg"] = metrics["l_ce_term_avg"] / metrics["l_ce_mask_avg"]
    return train_state, metrics


def train_loop():
    if FLAGS.wb_enabled and jax.process_index() == 0:
        logging.info("Creating W&B connection...")
        wandb.init(
            project=get_model_name(),
            config=vars(FLAGS.config)["_fields"],
            resume="never" if FLAGS.wb_run is None else "must",
            id=FLAGS.wb_run,
        )

    logging.info("Creating checkpoint manager...")
    checkpoint_mgr = get_checkpoint_manager()
    start_step = checkpoint_mgr.latest_step() or 0

    logging.info("Creating dataset...")
    dataset = get_dataset()
    train_iter = dataset.get_iter(
        split_name="train",
        batch_size=FLAGS.config.global_batch_size // jax.process_count(),
        sequence_len=FLAGS.config.sequence_len,
        shuffle_seed_root=start_step,  # with preemptions, picks a new shuffle
    )

    logging.info("Creating train state...")
    # always init and replicate inside the function doing reassigning of train_state,
    # otherwise there will be a dangling ref to a second (and non-updated) train_state,
    # which wastes a lot of memory when trying to train large models!
    rng_sync, rng_unsync = get_root_rngs()
    train_state = get_train_state(rng_sync)
    if checkpoint_mgr.latest_step():
        train_state = do_restore(checkpoint_mgr, train_state)
    train_state = jax.block_until_ready(jax_utils.replicate(train_state))

    logging.info("Starting training loop...")
    val_metrics = dict()
    best_val_loss = float("inf")
    start_time = time.perf_counter()
    for step in range(start_step, FLAGS.config.n_max_step + 1):
        train_state, metrics = train_step(
            train_state=train_state,
            batch=common_utils.shard(next(train_iter)),
            rng=common_utils.shard_prng_key(jax.random.fold_in(rng_unsync, step)),
        )
        if step % FLAGS.config.n_print_step == 0:
            metrics = jax_utils.unreplicate(metrics)
            metrics = jax.block_until_ready(metrics)
            end_time = time.perf_counter()
            metrics = dict(
                **metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                step=step,
                secs_per_step=(end_time - start_time) / FLAGS.config.n_print_step,
            )
            logging.info("-" * 80)
            for k, v in metrics.items():
                logging.info(f"\t{k}: {v.item() if isinstance(v, jax.Array) else v}")
            if FLAGS.wb_enabled and jax.process_index() == 0:
                wandb.log(metrics)
            start_time = end_time
        if step % FLAGS.config.n_save_step == 0:
            val_metrics = eval_loop(
                train_state=train_state,
                checkpoint_mgr=None,
                split_name="validation",
                n_eval_step=FLAGS.config.n_eval_step,
            )
            last_val_loss = val_metrics["l_ce_avg"].item()
            if best_val_loss > last_val_loss:
                logging.info("val loss improved")
                if best_val_loss < float("inf"):
                    logging.info("saving checkpoint")
                    do_save(
                        mgr=checkpoint_mgr,
                        target=jax_utils.unreplicate(train_state),
                        step=step,
                    )
                best_val_loss = last_val_loss
    return train_state


@functools.partial(jax.pmap, axis_name="devices")
def eval_step(params, batch):
    config = get_transformer_config(is_train=False)
    _, metrics = loss_fn(params, config, batch, rng=None)
    return jax.lax.pmean(metrics, axis_name="devices")


def eval_loop(
    split_name: str,
    train_state: Optional[train_utils.TrainState] = None,
    checkpoint_mgr: Optional[ocp.CheckpointManager] = None,
    n_eval_step: Optional[int] = None,
):
    logging.info("Creating dataset...")
    local_batch_size = FLAGS.config.global_batch_size // jax.process_count()
    dataset = get_dataset()
    eval_iter = dataset.get_iter(
        split_name=split_name,
        batch_size=local_batch_size,
        sequence_len=FLAGS.config.sequence_len,
        shuffle_seed_root=None,
    )

    if train_state is None:
        logging.info("Creating checkpoint manager...")
        checkpoint_mgr = checkpoint_mgr or get_checkpoint_manager()
        logging.info("Creating train state...")
        rng_sync, rng_unsync = get_root_rngs()
        train_state = get_train_state(rng_sync)
        if checkpoint_mgr.latest_step():
            train_state = do_restore(checkpoint_mgr, train_state)
        train_state = jax.block_until_ready(jax_utils.replicate(train_state))

    start_time = time.perf_counter()
    accumulator = None
    for i, batch in enumerate(eval_iter):
        logging.info(f"eval step {i}...")
        stats = eval_step(
            params=train_state.params,
            batch=common_utils.shard(batch),
        )
        stats = jax.block_until_ready(stats)
        stats = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), stats)
        if accumulator is not None:
            accumulator = jax.tree_util.tree_map(lambda a, b: a + b, stats, accumulator)
        else:
            accumulator = stats
        if n_eval_step is not None:
            if i + 1 == n_eval_step:
                break

    l_ce_avg = accumulator["l_ce_term_avg"]
    l_ce_avg /= accumulator["l_ce_mask_avg"]
    # calculation below assumes eval_op averages over global batch tokens
    # and the eval data is *replicated* over hosts
    # and the eval data is *sharded* over devices per host.
    #
    # if eval_data is sharded over hosts,
    #     you should multiply total_tokens by a further factor of jax.process_count()
    mult = local_batch_size * FLAGS.config.sequence_len
    total_tokens = mult * accumulator["l_ce_mask_avg"]
    eval_metrics = dict(l_ce_avg=l_ce_avg, total_tokens=total_tokens)
    eval_metrics = jax.block_until_ready(jax_utils.unreplicate(eval_metrics))
    end_time = time.perf_counter()
    eval_metrics.update(dict(secs_per_step=(end_time - start_time) / i))
    return eval_metrics


def flop_count():
    global_batch_size = FLAGS.config.global_batch_size
    sequence_len = FLAGS.config.sequence_len
    local_batch_size = global_batch_size // jax.process_count()
    setattr(FLAGS.config, "n_vocab", 256)

    rng_sync, rng_unsync = get_root_rngs()
    train_state = get_train_state(rng_sync)
    train_state = jax_utils.replicate(train_state)

    inputs = jax.random.randint(
        key=jax.random.fold_in(rng_unsync, hash("inputs")),
        minval=0,
        maxval=256,
        shape=[local_batch_size, sequence_len],
        dtype=jnp.int32,
    )
    targets = jax.random.randint(
        key=jax.random.fold_in(rng_unsync, hash("targets")),
        minval=0,
        maxval=256,
        shape=[local_batch_size, sequence_len],
        dtype=jnp.int32,
    )
    batch = dict(
        inputs=inputs,
        targets=targets,
        loss_mask=jnp.full([local_batch_size, sequence_len], fill_value=1),
    )
    batch = common_utils.shard(batch)

    rng_batch = common_utils.shard_prng_key(rng_unsync)

    compiled = train_step.lower(train_state, batch, rng_batch).compile()
    cost_analysis = compiled.cost_analysis()
    logging.info("== Cost Analysis ==")
    if cost_analysis is None:
        logging.info("Cost analysis is not available from compiler on platform.")
    else:
        logging.info(f"Num hosts: {jax.process_count()}")
        logging.info(f"Num devices per host: {jax.local_device_count()}")
        n_flop = cost_analysis[0]["flops"]
        logging.info(f"FLOP count estimate: {n_flop}")
        n_param = sum(
            x.size
            for x in jax.tree_util.tree_leaves(jax_utils.unreplicate(train_state))
        )
        logging.info(f"Param count: {n_param}")
        n_example_per_dev = (local_batch_size / jax.local_device_count()) * sequence_len
        logging.info(f"Example count per host: {n_example_per_dev}")
        n_min_flop_reasonable = n_example_per_dev * n_param
        logging.info(f"Reasonable flop count min: {n_min_flop_reasonable}")
        is_reasonable = n_flop >= n_min_flop_reasonable
        logging.info(f"FLOP count estimate is reasonable: {is_reasonable}")


def log_devices_and_config():
    logging.info("== Devices ==")
    logging.info(jax.devices())

    logging.info("== Local Devices ==")
    logging.info(jax.local_devices())

    logging.info("== Launch config ==")
    logging.info(f"multihost: {FLAGS.multihost}")
    logging.info(f"workdir: {FLAGS.workdir}")
    logging.info(f"mode: {FLAGS.mode}")
    logging.info(f"wb_enabled: {FLAGS.wb_enabled}")
    logging.info(f"wb_run: {FLAGS.wb_run}")

    logging.info("== Model config ==")
    for k, v in vars(FLAGS.config)["_fields"].items():
        logging.info(f"{k}: {v}")


def main(argv):
    del argv  # unused
    tf.get_logger().setLevel(logging.INFO)
    logging.set_verbosity(logging.INFO)
    if FLAGS.multihost:
        if FLAGS.coordinator_addr is None:
            jax.distributed.initialize()
        else:
            jax.distributed.initialize(
                coordinator_address=FLAGS.gpu_coord_addr,
                num_processes=FLAGS.gpu_n_process,
                process_id=FLAGS.gpu_process_id,
            )
    if jax.default_backend() in {"cuda", "rocm"}:
        # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
        # it unavailable to JAX. (Not necessary with TPU.)
        tf.config.experimental.set_visible_devices([], "GPU")

    log_devices_and_config()
    if FLAGS.mode in {"train_vocab", "train"}:
        train_loop()
    elif FLAGS.mode in {"validation", "test"}:
        eval_metrics = eval_loop(split_name=FLAGS.mode)
        logging.info(eval_metrics)
    elif FLAGS.mode == "flop_count":
        flop_count()
    else:
        raise ValueError(f"Operation {FLAGS.mode} not implemented in main.")


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
