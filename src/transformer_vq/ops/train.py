import functools

import jax
import jax.numpy as jnp

from transformer_vq.nn.grad import sg
from transformer_vq.nn.model import Transformer
from transformer_vq.nn.vq import VQSpec
from transformer_vq.ops.loss import loss_fn


@functools.partial(
    jax.pmap,
    axis_name="devices",
    static_broadcasted_argnums=(0,),
    donate_argnums=(1,),
)
def train_op(config, train_state, batch, rng):
    n_device = jax.device_count()
    n_update = config.sequence_len // config.update_len
    n_block_per_update = config.update_len // config.block_len
    device_batch_size = config.global_batch_size // n_device
    rng, rng_timeless = jax.random.split(rng)

    # used by outer scan to run multiple parameter updates for a long sequence
    def update_body(carry, input_dict):
        rng_new, rng_ephemeral = jax.random.split(carry["rng"])
        (_, aux), grads_update = jax.value_and_grad(loss_fn, has_aux=True)(
            carry["train_state"].params,
            config=config,
            batch=input_dict,
            attn_state=carry["attn_state"],
            vq_spec=VQSpec.create(
                n_device=jnp.array([n_device]),
                n_block_per_update=jnp.array([n_block_per_update]),
                loss_mask=input_dict["loss_mask"],
            ),
            rngs=dict(ephemeral=rng_ephemeral, timeless=rng_timeless),
        )
        grads_update = jax.lax.pmean(grads_update, axis_name="devices")
        metrics_update = jax.lax.pmean(aux["metrics"], axis_name="devices")
        carry_new = dict(
            attn_state=jax.tree_util.tree_map(sg, aux["attn_state"]),
            train_state=carry["train_state"].apply_gradients(grads=grads_update),
            rng=rng_new,
        )
        return carry_new, metrics_update

    def do_reshape(tensor):
        shape = [device_batch_size, n_update, n_block_per_update * config.block_len]
        tensor = jnp.reshape(tensor, shape)
        tensor = jnp.transpose(tensor, (1, 0, 2))
        return tensor

    outer_carry_final, metrics_all = jax.lax.scan(
        f=update_body,
        init=dict(
            attn_state=Transformer.initial_state(config, device_batch_size),
            train_state=train_state,
            rng=rng,
        ),
        xs=jax.tree_util.tree_map(do_reshape, batch),
        length=n_update,
        unroll=1,
    )
    metrics_all = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics_all)
    new_train_state = outer_carry_final["train_state"]
    return new_train_state, metrics_all
