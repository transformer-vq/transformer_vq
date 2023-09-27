import jax.numpy as jnp

from transformer_vq.nn.grad import sg
from transformer_vq.nn.model import Transformer


def loss_fn(
    params,
    config,
    batch,
    attn_state,
    vq_spec,
    rngs,
):
    call_kwargs = dict(
        inputs=batch["inputs"],
        doc_ids=batch["doc_ids"],
        state=attn_state,
        vq_spec=vq_spec,
    )
    if rngs is not None:
        call_kwargs["rngs"] = rngs
    outputs = Transformer(config).apply({"params": params}, **call_kwargs)
    # averages loss_lm over each device's batch positions and segment tokens
    logprobs = outputs["logprobs"]
    l_lm_premask = -jnp.take_along_axis(logprobs, batch["targets"][..., None], axis=-1)
    l_lm_premask = jnp.squeeze(l_lm_premask, axis=-1)
    l_lm_unscaled = (batch["loss_mask"] * l_lm_premask).mean()
    zero = jnp.zeros([], dtype=jnp.float32)
    l_commit = outputs["l_commit"] if "l_commit" in outputs else zero
    l_codebook = outputs["l_codebook"] if "l_codebook" in outputs else zero
    outputs["l_lm_unscaled"] = l_lm_unscaled
    composite_loss = l_lm_unscaled + config.c_beta * l_commit + l_codebook
    if "metrics" not in outputs:
        outputs["metrics"] = dict()
    outputs["metrics"]["loss_lm_per_token_unscaled"] = sg(l_lm_unscaled)
    outputs["metrics"]["loss_mask_per_token"] = sg(batch["loss_mask"].mean())
    outputs["metrics"]["loss_commit_per_token"] = sg(l_commit)
    return composite_loss, outputs
