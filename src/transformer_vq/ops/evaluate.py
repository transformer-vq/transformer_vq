import functools

import jax
import jax.numpy as jnp

from transformer_vq.nn.model import Transformer
from transformer_vq.ops.loss import loss_fn


@functools.partial(
    jax.pmap,
    axis_name="devices",
    static_broadcasted_argnums=(0,),
)
def eval_op(config, params, batch):
    device_batch_size = config.global_batch_size // jax.device_count()
    n_block = config.sequence_len // config.block_len

    # loop here for memory efficiency: big models give oom on long sequences without it
    def body(carry, input_dict):
        _, fwd_dict = loss_fn(
            params,
            config=config,
            batch=input_dict,
            attn_state=carry,
            vq_spec=None,
            rngs=None,
        )
        carry_new = fwd_dict["attn_state"]
        metrics = fwd_dict["metrics"]
        block_stats_local = dict(
            loss_lm_per_token_unscaled=metrics["loss_lm_per_token_unscaled"],
            loss_mask_per_token=metrics["loss_mask_per_token"],
        )
        block_stats_global = jax.lax.pmean(block_stats_local, axis_name="devices")
        return carry_new, block_stats_global

    def do_reshape(tensor):
        tensor = jnp.reshape(tensor, [device_batch_size, n_block, config.block_len])
        tensor = jnp.transpose(tensor, (1, 0, 2))
        return tensor

    _, multiblock_stats_global = jax.lax.scan(
        f=body,
        init=Transformer.initial_state(config, device_batch_size),
        xs=jax.tree_util.tree_map(do_reshape, batch),
        length=n_block,
        unroll=1,
    )
    sequence_stats_global = jax.tree_util.tree_map(
        lambda y: jnp.mean(y, axis=0), multiblock_stats_global
    )
    return sequence_stats_global
