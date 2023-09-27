import jax
import jax.numpy as jnp
import numpy as np

from transformer_vq.nn.grad import sg
from transformer_vq.nn.grad import st


def test_sg():
    input_ = jax.random.normal(jax.random.PRNGKey(0), [10], dtype=jnp.float32)
    np.testing.assert_allclose(actual=sg(input_), desired=input_)
    np.testing.assert_allclose(
        actual=jax.jacobian(sg)(input_), desired=np.zeros([10, 10])
    )


def test_st():
    input_ = jax.random.normal(jax.random.PRNGKey(0), [10], dtype=jnp.float32)
    np.testing.assert_allclose(actual=st(input_), desired=np.zeros_like(input_))
    np.testing.assert_allclose(actual=jax.jacobian(st)(input_), desired=np.eye(10))
