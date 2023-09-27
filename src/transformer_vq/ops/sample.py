import functools

import jax
import jax.numpy as jnp

from transformer_vq.nn.model import Transformer
from transformer_vq.nn.prob import nucleus


@functools.partial(
    jax.pmap,
    axis_name="devices",
    static_broadcasted_argnums=(0, 1),
)
def sample_op(config, eos_id, params, rng):
    device_batch_size = config.global_batch_size // jax.device_count()

    def body(carry, discard):
        rng_new, rng_sample = jax.random.split(carry["rng"])
        outputs = Transformer(config).apply(
            {"params": params},
            inputs=carry["token_prev"],
            doc_ids=carry["doc_ids"],
            state=carry["attn_state"],
            vq_spec=None,
            rngs=None,
        )
        nucleus_logits = nucleus(outputs["logprobs"][:, -1:, :], p=config.p_nucleus)
        token_new = jax.random.categorical(rng_sample, logits=nucleus_logits)
        carry_new = dict(
            attn_state=outputs["attn_state"],
            token_prev=token_new,
            doc_ids=carry["doc_ids"] + jnp.equal(token_new, eos_id).astype(jnp.int32),
            rng=rng_new,
        )
        return carry_new, token_new

    _, tokens_all = jax.lax.scan(
        f=body,
        init=dict(
            attn_state=Transformer.initial_state(config, device_batch_size),
            doc_ids=jnp.zeros(shape=[device_batch_size, 1], dtype=jnp.int32),
            token_prev=eos_id * jnp.ones(shape=[device_batch_size, 1], dtype=jnp.int32),
            rng=rng,
        ),
        xs=jnp.arange(config.sequence_len),
        length=config.sequence_len,
        unroll=1,
    )
    tokens_all = jnp.squeeze(tokens_all, -1)
    tokens_all = jnp.transpose(tokens_all, (1, 0))
    return tokens_all
