from typing import Dict
from typing import Tuple
from typing import Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax.training import common_utils
from flax.training.train_state import TrainState

FLAGS = flags.FLAGS
flags.DEFINE_boolean("use_remat_scan", True, "Use remat_scan?")

BSZ = 8
D_MODEL = 10
D_FF = 40
N_LAYER = 24


class MLP(nn.Module):
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        af1 = nn.Dense(self.d_ff)(x)
        af2 = nn.Dense(self.d_model)(jax.nn.silu(af1))
        return x + af2


class StackedMLPsRematted(nn.Module):
    d_model: int
    d_ff: int
    n_layer: int

    @nn.compact
    def __call__(self, x):
        return nn.remat_scan(MLP, lengths=(self.n_layer, 1))(
            d_model=self.d_model,
            d_ff=self.d_ff,
            name="stack",
        )(x)


class StackedMLPsListed(nn.Module):
    d_model: int
    d_ff: int
    n_layer: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layer):
            x = MLP(d_model=self.d_model, d_ff=self.d_ff)(x)
        return x


def loss_fn(params, batch, cls):
    predictions = cls(D_MODEL, D_FF, N_LAYER).apply({"params": params}, batch["inputs"])
    loss_terms = optax.l2_loss(targets=batch["targets"], predictions=predictions)
    loss = jnp.mean(loss_terms)
    return loss


def train_op(
    train_state: TrainState,
    batch: Dict[str, jax.Array],
    is_pmapped: bool,
    cls: Type[nn.Module],
) -> Tuple[TrainState, jax.Array]:
    loss, grads = jax.value_and_grad(loss_fn)(
        train_state.params,
        batch=batch,
        cls=cls,
    )
    if is_pmapped:
        loss, grads = jax.lax.pmean([loss, grads], axis_name="devices")
    return train_state.apply_gradients(grads=grads), loss


def get_deterministic_trainstate_and_batch(cls):
    x = jax.random.normal(jax.random.PRNGKey(0), [BSZ, D_MODEL])
    y = jax.random.normal(jax.random.PRNGKey(1), [BSZ, D_MODEL])
    batch = dict(inputs=x, targets=y)
    ps = cls(D_MODEL, D_FF, N_LAYER).init({"params": jax.random.PRNGKey(2)}, x)
    train_state = TrainState.create(
        apply_fn=None,
        params=ps["params"].unfreeze(),
        tx=optax.sgd(learning_rate=0.01),
    )
    return train_state, batch


def pmapped_train_op_cost_analysis(cls):
    train_state, batch = get_deterministic_trainstate_and_batch(cls)
    p_train_op = jax.pmap(
        train_op,
        axis_name="devices",
        donate_argnums=(0,),
        static_broadcasted_argnums=(2, 3),
    )
    compiled = p_train_op.lower(
        jax_utils.replicate(train_state),
        common_utils.shard(batch),
        True,
        cls,
    ).compile()
    cost_analysis = compiled.cost_analysis()
    return cost_analysis


def jitted_train_op_cost_analysis(cls):
    train_state, batch = get_deterministic_trainstate_and_batch(cls)
    j_train_op = jax.jit(train_op, donate_argnums=(0,), static_argnums=(2, 3))
    compiled = j_train_op.lower(
        train_state,
        batch,
        False,
        cls,
    ).compile()
    cost_analysis = compiled.cost_analysis()
    return cost_analysis


def evaluate_cost_analysis(train_state, cost_analysis):
    if cost_analysis is not None:
        n_flop = cost_analysis[0]["flops"]
        logging.info(f"FLOP count estimate: {n_flop}")
        n_param = sum(x.size for x in jax.tree_util.tree_leaves(train_state.params))
        logging.info(f"Param count: {n_param}")
        n_example = BSZ / jax.local_device_count()
        logging.info(f"Example count: {n_example}")
        n_min_flop_reasonable = n_example * n_param
        logging.info(f"Reasonable flop count min: {n_min_flop_reasonable}")
        is_reasonable = n_flop > n_min_flop_reasonable
        logging.info(f"Is reasonable estimate: {is_reasonable}")


def main(argv):
    del argv
    cls = StackedMLPsRematted if FLAGS.use_remat_scan else StackedMLPsListed

    cost_for_pmapped_train_op = pmapped_train_op_cost_analysis(cls)
    train_state, _ = get_deterministic_trainstate_and_batch(cls)
    evaluate_cost_analysis(train_state, cost_for_pmapped_train_op)

    cost_for_jitted_train_op = jitted_train_op_cost_analysis(cls)
    train_state, _ = get_deterministic_trainstate_and_batch(cls)
    evaluate_cost_analysis(train_state, cost_for_jitted_train_op)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
