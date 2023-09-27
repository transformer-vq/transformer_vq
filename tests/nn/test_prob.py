import jax
import jax.numpy as jnp
import numpy as np

from transformer_vq.nn.prob import nucleus


def test_argsort_inversion() -> None:
    x = jax.random.normal(jax.random.PRNGKey(0), shape=[100, 10], dtype=jnp.float32)
    sorted_x = jnp.sort(x, axis=-1)
    sort_indices = jnp.argsort(x, axis=-1)
    unsort_indices = jnp.argsort(sort_indices, axis=-1)
    x_ = jnp.take_along_axis(sorted_x, unsort_indices, axis=-1)
    np.testing.assert_allclose(actual=x_, desired=x)


def test_nucleus() -> None:
    probs = jnp.array(
        [0.05, 0.05, 0.05, 0.09, 0.11, 0.15, 0.30, 0.20], dtype=jnp.float32
    )
    logits = jnp.log(probs)

    actual = nucleus(logits=logits, p=0.25)
    expected = jnp.array([-1e10] * 6 + [logits[-2]] + [-1e10])
    np.testing.assert_allclose(actual=actual, desired=expected)

    actual = nucleus(logits=logits, p=0.31)
    expected = jnp.concatenate([jnp.array([-1e10] * 6), logits[-2:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    actual = nucleus(logits=logits, p=0.65)
    expected = jnp.concatenate([jnp.array([-1e10] * 5), logits[-3:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    actual = nucleus(logits=logits, p=0.75)
    expected = jnp.concatenate([jnp.array([-1e10] * 4), logits[-4:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    actual = nucleus(logits=logits, p=0.85)
    expected = jnp.concatenate([jnp.array([-1e10] * 3), logits[-5:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    actual = nucleus(logits=logits, p=0.90)
    expected = jnp.concatenate([jnp.array([-1e10] * 2), logits[-6:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    actual = nucleus(logits=logits, p=0.95)
    expected = jnp.concatenate([jnp.array([-1e10] * 1), logits[-7:]], axis=-1)
    np.testing.assert_allclose(actual=actual, desired=expected)

    actual = nucleus(logits=logits, p=0.99)
    expected = logits
    np.testing.assert_allclose(actual=actual, desired=expected)

    actual = nucleus(logits=logits, p=1.0)
    expected = logits
    np.testing.assert_allclose(actual=actual, desired=expected)
