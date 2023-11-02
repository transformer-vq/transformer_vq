import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import jax_utils
from flax.training import common_utils
from flax.training.train_state import TrainState

from transformer_vq.nn.model import Transformer
from transformer_vq.nn.types import TransformerConfig
from transformer_vq.nn.vq import VQSpec
from transformer_vq.ops.evaluate import eval_op
from transformer_vq.ops.sample import sample_op
from transformer_vq.ops.train import train_op
from transformer_vq.utils.datasets import Dataset
from transformer_vq.utils.io import load_checkpoint
from transformer_vq.utils.io import save_checkpoint
from transformer_vq.utils.io import save_pixels
from transformer_vq.utils.io import save_text
from transformer_vq.utils.tree import flattened_traversal


DTYPES = ["bfloat16", "float32"]
COMMANDS = ["train_vocab", "train", "validation", "test", "sample", "bench"]
DATASETS = ["enwik8", "pg19", "imagenet64"]
OPTIMIZERS = ["adamw", "lion", "adafactor"]

parser = argparse.ArgumentParser("Launch script for Transformer VQ experiments.")
parser.add_argument("--multihost", type=int, help="Multihost mode?", default=0)
parser.add_argument("--command", choices=COMMANDS)
parser.add_argument("--dataset", choices=DATASETS)
parser.add_argument("--data_dir", type=str, help="Download path", default=None)
parser.add_argument("--vocab_path", type=str, help="Sentencepiece path", default=None)
parser.add_argument("--prng_seed", type=int, help="PRNG seed")
parser.add_argument("--global_batch_size", type=int, help="Global batch size")
parser.add_argument("--sequence_len", type=int, help="Sequence length T")
parser.add_argument("--update_len", type=int, help="Update length LK")
parser.add_argument("--block_len", type=int, help="Block length L")
parser.add_argument("--mem_len", type=int, help="Band length M")
parser.add_argument("--grad_thru_cache", type=int, help="Backprop thru cache (0/1)")
parser.add_argument("--agg_cache", type=int, help="Include aggregated cache (0/1)")

parser.add_argument("--param_dtype", choices=DTYPES, help="Dtype for parameters")
parser.add_argument("--dtype", choices=DTYPES, help="Dtype for computation")
parser.add_argument("--d_model", type=int, help="Model width")
parser.add_argument("--d_k", type=int, help="Key width")
parser.add_argument("--d_v", type=int, help="Value width")
parser.add_argument("--d_ff", type=int, help="Fan-out width, if using MLPs", default=0)
parser.add_argument("--n_head", type=int, help="Num attention heads")
parser.add_argument("--n_code", type=int, help="Num codes per head")
parser.add_argument("--n_layer", type=int, help="Num transformer layers (two GAU each)")
parser.add_argument("--pe_abs", type=int, help="Include abs pos embs (0/1)")
parser.add_argument("--pe_lam", type=int, help="Max angular wavelength", default=100000)
parser.add_argument("--p_dropemb", type=float, help="Embedding dropout rate")
parser.add_argument("--p_dropsin", type=float, help="Rel PE sinusoid dropout rate")
parser.add_argument("--p_dropres", type=float, help="Residual dropout rate")
parser.add_argument("--p_droplyr", type=float, help="LayerDrop rate")
parser.add_argument("--c_beta", type=float, help="Codebook commit coefficient")
parser.add_argument("--c_gamma", type=float, help="Codebook EMA rate")
parser.add_argument("--e_tie", type=int, help="Output embs tied w input embs (0/1)")
parser.add_argument("--e_preln", type=int, help="Output embs applied after LN (0/1)")
parser.add_argument("--e_scale", type=float, help="Output embs scale factor")

parser.add_argument("--grad_clip", type=float, help="Gradient clip norm", default=None)
parser.add_argument("--optimizer", choices=OPTIMIZERS, help="Optimizer name")
parser.add_argument("--lr_max", type=float, help="Peak learning rate")
parser.add_argument("--lr_schedule", type=str, help="Learning rate schedule name")
parser.add_argument("--wd_lam", type=float, help="Decoupled weight decay")
parser.add_argument("--p_nucleus", type=float, help="Nucleus cutoff during sampling")
parser.add_argument("--n_warmup_step", type=int, help="Linear warmup steps")
parser.add_argument("--n_max_step", type=int, help="Maximum step number")
parser.add_argument("--n_extra_step", type=int, help="Extra steps, use > 0 in finetune")
parser.add_argument("--n_print_step", type=int, help="Steps per print", default=100)
parser.add_argument("--n_save_step", type=int, help="Train steps between eval phases")
parser.add_argument("--n_eval_step", type=int, help="Batches per eval phase")
parser.add_argument("--n_save_keep", type=int, help="Checkpoints to keep", default=5)
parser.add_argument("--in_checkpoint_dir", type=str, help="Checkpoint dir to load from")
parser.add_argument("--out_checkpoint_dir", type=str, help="Checkpoint dir to save to")
parser.add_argument("--model_name", type=str, help="Model name")
parser.add_argument("--run_id", type=str, help="For logging continuity", default=None)
args = parser.parse_args()


if args.multihost:
    jax.distributed.initialize()


def print_mem_info():
    backend = jax.lib.xla_bridge.get_backend()
    n_bufs = len(backend.live_buffers())

    def tobytes(b):
        return np.prod(b.shape) * int(str(b.dtype)[-2:]) // 8

    n_bytes = sum([tobytes(b) for b in backend.live_buffers()])
    print(f"num_live_buffers: {n_bufs}")
    print(f"num_live_bytes: {n_bytes}")
    for i, buf in enumerate(backend.live_buffers()):
        # correct number of printed items depends on optimizer and whether embs are tied
        if args.n_vocab in list(buf.shape):
            print(f"buffer_{i}.shape: {buf.shape}")


def get_param_label_fn():
    return flattened_traversal(
        lambda path, _: "main" if not path[-1].startswith("c_") else "codebook"
    )


def get_schedule_fn():
    warmup = optax.linear_schedule(0.0, 1.0, transition_steps=args.n_warmup_step)
    dec_steps = args.n_max_step - args.n_warmup_step  # exclude extra, const finetune lr
    if args.lr_schedule == "cosine":
        main_schedule = optax.cosine_decay_schedule(
            init_value=1.0,
            decay_steps=dec_steps,
            alpha=0.1,  # final schedule value
        )
    elif args.lr_schedule == "inv_sqrt":
        main_schedule = optax.polynomial_schedule(
            init_value=1.0,
            end_value=0.1,
            power=-0.5,
            transition_steps=dec_steps,
        )
    elif args.lr_schedule == "linear":
        main_schedule = optax.linear_schedule(
            init_value=1.0,
            end_value=0.1,
            transition_steps=dec_steps,
        )
    else:
        raise NotImplementedError("schedule name not supported")
    return optax.join_schedules(
        schedules=[warmup, main_schedule], boundaries=[args.n_warmup_step]
    )


def get_optimizer():
    if args.optimizer == "adamw":
        return optax.adamw(
            learning_rate=args.lr_max,
            b1=0.9,
            b2=0.98,
            eps=10**-9,
            mu_dtype=jnp.float32,  # full precision as suggested by Rae et al., 2021
            weight_decay=0.0,  # optimizers in optax scale wd by lr, so diy
        )
    if args.optimizer == "lion":
        return optax.lion(
            learning_rate=args.lr_max,
            b1=0.95,
            b2=0.98,
            mu_dtype=jnp.bfloat16,  # bfloat16 as suggested by Chen et al., 2023
            weight_decay=0.0,  # optimizers in optax scale wd by lr, so diy
        )
    if args.optimizer == "adafactor":
        return optax.adafactor(
            learning_rate=args.lr_max,
            multiply_by_parameter_scale=True,
            clipping_threshold=1.0,  # must be >= 1.0 per optax docs.
            weight_decay_rate=0.0,  # optimizers in optax scale wd by lr, so diy
        )


def get_tx():
    maybe_gradclip = []
    if args.grad_clip is not None:
        maybe_gradclip.append(optax.clip_by_global_norm(args.grad_clip))
    optimizer = get_optimizer()
    schedule = optax.scale_by_schedule(get_schedule_fn())
    maybe_decay = []
    if args.wd_lam > 0.0:
        wd = optax.add_decayed_weights(
            weight_decay=-args.wd_lam,
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


def get_train_state(init_rng):
    config = dict(**vars(args), is_train=True)
    _ = config.pop("block_len")
    config["block_len"] = 8
    config = TransformerConfig.create(**config)
    model = Transformer(config)
    sk1, sk2, sk3, sk4 = jax.random.split(init_rng, 4)
    rngs = dict(params=sk1, ephemeral=sk2, timeless=sk3)
    inputs = jnp.zeros([1, config.block_len], dtype=jnp.int32)
    doc_ids = jnp.zeros([1, config.block_len], dtype=jnp.int32)
    state = Transformer.initial_state(config=config, batch_size=1)
    vq_spec = VQSpec.create(
        n_device=jnp.array([1]),
        n_block_per_update=jnp.array([1]),
        loss_mask=jnp.ones([1, config.block_len], jnp.int32),
    )
    params = model.init(
        rngs,
        inputs=inputs,
        doc_ids=doc_ids,
        state=state,
        vq_spec=vq_spec,
    )["params"].unfreeze()
    tx = get_tx()
    if args.command != "bench":
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"param count: {param_count}")
    return TrainState.create(apply_fn=None, params=params, tx=tx)


def train(dataset, p_train_op):
    assert args.sequence_len % args.update_len == 0
    assert args.update_len % args.block_len == 0
    assert args.sequence_len % args.block_len == 0
    assert args.n_save_step >= (args.sequence_len // args.update_len)
    assert args.n_save_step % (args.sequence_len // args.update_len) == 0

    # always init and replicate inside the function doing reassigning of train_state,
    # otherwise there will be a dangling ref to a second (and non-updated) train_state,
    # which wastes a lot of memory when trying to train large models!
    train_state, rng, start_step = state_setup()
    train_state = jax.block_until_ready(jax_utils.replicate(train_state))
    local_batch_size = args.global_batch_size // jax.process_count()
    n_update = args.sequence_len // args.update_len

    train_config = TransformerConfig.create(**vars(args), is_train=True)
    train_iter = dataset.get_iter(
        split_name="train",
        batch_size=local_batch_size,
        sequence_len=args.sequence_len,
    )

    val_metrics = dict()
    best_val_loss_lm = float("inf")
    start_time = time.perf_counter()
    for step in range(start_step, args.n_max_step + args.n_extra_step + 1, n_update):
        if step % args.n_save_step == 0:
            val_metrics = evaluate(
                train_state=train_state,
                dataset=dataset,
                split_name="validation",
                p_eval_op=eval_op,
                n_eval_step=args.n_eval_step,  # per host
                persist=False,
            )
            last_val_loss_lm = val_metrics["loss_lm_per_token"].tolist()
            if best_val_loss_lm > last_val_loss_lm:
                best_val_loss_lm = last_val_loss_lm
                print("val loss improved")
                if step > start_step:
                    print("saving checkpoint")
                    save_checkpoint(
                        target=jax_utils.unreplicate(train_state),
                        save_dir=args.out_checkpoint_dir,
                        prefix="checkpoint",
                        step=step,
                        keep=args.n_save_keep,
                    )
        batch = next(train_iter)
        rng, batch_rng = jax.random.split(rng)
        train_state, metrics = p_train_op(
            train_config,
            train_state=train_state,
            batch=common_utils.shard(batch),
            rng=common_utils.shard_prng_key(batch_rng),
        )
        if step % args.n_print_step == 0:
            print_mem_info()
            metrics = jax_utils.unreplicate(metrics)
            metrics = jax.block_until_ready(metrics)
            train_loss_lm_per_token = metrics["loss_lm_per_token_unscaled"]
            train_loss_lm_per_token /= metrics["loss_mask_per_token"]
            end_time = time.perf_counter()
            logging_info = dict(
                loss_lm_per_token=train_loss_lm_per_token,
                **metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                step=step,
                secs_per_step=(end_time - start_time) / n_update,
            )
            print(train_state.step)
            print(batch["inputs"].shape)
            print(batch["targets"].shape)
            print(logging_info)
            if jax.process_index() == 0:
                wandb.log(logging_info)
            start_time = end_time
    return train_state


def evaluate(
    train_state,
    dataset,
    split_name,
    p_eval_op,
    n_eval_step=None,
    persist=False,
):
    assert args.sequence_len % args.block_len == 0
    if train_state is None:
        train_state, rng, start_step = state_setup()
        train_state = jax.block_until_ready(jax_utils.replicate(train_state))
    step = int(jax_utils.unreplicate(train_state.step))
    local_batch_size = args.global_batch_size // jax.process_count()

    eval_config = TransformerConfig.create(**vars(args), is_train=False)
    eval_iter = dataset.get_iter(
        split_name=split_name,
        batch_size=local_batch_size,
        sequence_len=args.sequence_len,
    )

    accumulator = None
    for i, batch in enumerate(eval_iter):
        print(f"eval step {i}...")
        stats = p_eval_op(
            eval_config,
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

    loss_lm_per_token = accumulator["loss_lm_per_token_unscaled"]
    loss_lm_per_token /= accumulator["loss_mask_per_token"]
    # calculation below assumes eval_op averages over global batch, blocks, tokens
    # and that the eval data is replicated over hosts and sharded over devices per host.
    mult = local_batch_size * args.sequence_len
    total_tokens = mult * accumulator["loss_mask_per_token"]
    eval_metrics = dict(loss_lm_per_token=loss_lm_per_token, total_tokens=total_tokens)
    eval_metrics = jax.block_until_ready(jax_utils.unreplicate(eval_metrics))
    if persist:
        save_kwargs = dict(
            prefix=f"{split_name}_loss_lm_per_token",
            save_dir=args.out_checkpoint_dir,
            step=step,
            keep=args.n_save_keep,
        )
        save_checkpoint(eval_metrics, **save_kwargs)
    return eval_metrics


def sample(dataset, p_sample_op, persist=False):
    train_state, rng, start_step = state_setup()
    step = int(train_state.step)
    train_state = jax.block_until_ready(jax_utils.replicate(train_state))
    local_batch_size = args.global_batch_size // jax.process_count()

    args_dict = vars(args)
    _ = args_dict.pop("block_len")
    sample_config = TransformerConfig.create(**args_dict, block_len=1, is_train=False)
    rng = common_utils.shard_prng_key(rng).block_until_ready()

    start_time = time.perf_counter()
    samples = p_sample_op(
        sample_config,
        dataset.vocab.eos_id,
        params=train_state.params,
        rng=rng,
    ).block_until_ready()  # block until ready so the time delta is correct
    end_time = time.perf_counter()

    total_time = end_time - start_time
    if persist:
        samples = jnp.reshape(samples, [local_batch_size, args.sequence_len])
        save_fn = dict(text=save_text, image=save_pixels)[dataset.modality]
        suffix = dict(text=".txt", image=".png")[dataset.modality]
        for i in range(local_batch_size):
            save_fn(
                target=dataset.decode(samples[i]),
                dirname=args.out_checkpoint_dir,
                fname=f"samples_step{step}_proc{jax.process_index()}_item{i}{suffix}",
            )
    return dict(total_time=total_time)


def print_args_and_devices():
    if args.command != "bench":
        print(jax.devices())
        print(jax.local_devices())
    if args.command != "bench":
        for k, v in vars(args).items():
            print(f"{k}: {v}")


def wandb_setup():
    if args.run_id == "":
        setattr(args, "run_id", None)
    if args.command == "train" and jax.process_index() == 0:
        wandb.init(
            project=args.model_name,
            config=vars(args),
            resume="never" if args.run_id is None else "must",
            id=args.run_id,
        )


def dataset_setup():
    dataset_cls = Dataset.registry.get(args.dataset)
    dataset = dataset_cls(vocab_path=args.vocab_path, data_dir=args.data_dir)
    if args.command == "train_vocab":
        sys.exit(0)
    setattr(args, "n_vocab", dataset.vocab_size)
    return dataset


def state_setup():
    synced_rng = jax.random.PRNGKey(args.prng_seed)
    synced_rng, init_rng = jax.random.split(synced_rng)
    train_state = get_train_state(init_rng)
    train_state = load_checkpoint(
        train_state=train_state,
        load_dir=args.in_checkpoint_dir,
        prefix="checkpoint",
    )
    start_step = train_state.step
    if args.command != "bench":
        print(f"start_step: {start_step}")
    synced_rng = jax.random.fold_in(synced_rng, start_step)
    unsynced_rng = jax.random.fold_in(synced_rng, jax.process_index())
    return train_state, unsynced_rng, start_step


def main():
    print_args_and_devices()
    wandb_setup()
    dataset = dataset_setup()

    if args.command == "train":
        train(dataset=dataset, p_train_op=train_op)

    elif args.command in {"validation", "test"}:
        eval_metrics = evaluate(
            train_state=None,
            dataset=dataset,
            split_name=args.command,
            p_eval_op=eval_op,
            persist=True,
        )
        print(eval_metrics)

    elif args.command in {"sample", "bench"}:
        sample_kwargs = dict(
            dataset=dataset,
            p_sample_op=sample_op,
        )
        if args.command == "sample":
            outputs = sample(**sample_kwargs, persist=True)
            print(outputs)
        else:
            # warm start for benchmarking to exclude JIT compile time of p_sample_op
            outputs = sample(**sample_kwargs, persist=False)  # cold started
            outputs = sample(**sample_kwargs, persist=False)  # warm started
            print(outputs["total_time"])

    else:
        raise ValueError(f"Operation {args.command} not implemented in main.")


if __name__ == "__main__":
    main()
