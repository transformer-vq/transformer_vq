import jax
import jax.numpy as jnp


def nucleus(logits, p):
    n_vocab = logits.shape[-1]
    # compute probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    # sort probabilities in ascending order and get their argsort indices.
    sorted_probs = jnp.sort(probs, axis=-1)
    sort_indices = jnp.argsort(probs, axis=-1)
    cumulative_sorted_probs = jnp.cumsum(sorted_probs, axis=-1)
    # create a nucleus mask for the sorted probabilities.
    # the mask accepts the largest probabilities whose sum is less than or equal to p,
    # and always includes the largest probability token.
    m1 = jnp.greater(cumulative_sorted_probs, 1.0 - (p - 1e-4))  # "is tail > 1-p"?
    m2 = jnp.equal(
        jnp.arange(n_vocab),
        jnp.full(fill_value=n_vocab - 1, shape=[n_vocab]),
    )
    mask_for_sorted = jnp.logical_or(m1, m2).astype(jnp.int32)
    # unsort the mask so that it applies to the token logits in their non-sorted order.
    unsort_indices = jnp.argsort(sort_indices, axis=-1)
    mask = jnp.take_along_axis(mask_for_sorted, unsort_indices, axis=-1)
    # mask out the non-nucleus logits.
    masked_logits = logits * mask - 1e10 * (1 - mask)
    return masked_logits
