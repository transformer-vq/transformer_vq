from dataclasses import fields
from typing import Any
from typing import Callable
from typing import List

import jax.nn.initializers as inits
import jax.numpy as jnp
from flax import struct
from jax import Array

PRNGKey = Any
Shape = List[int]
Dtype = Any
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]


@struct.dataclass
class TransformerConfig:
    param_dtype: Dtype
    dtype: Dtype
    global_batch_size: int
    sequence_len: int
    update_len: int
    block_len: int
    mem_len: int
    grad_thru_cache: bool
    agg_cache: bool
    d_model: int
    d_k: int
    d_v: int
    d_ff: int
    n_head: int
    n_code: int
    n_layer: int
    n_vocab: int
    pe_abs: bool
    pe_lam: float
    p_dropemb: float
    p_dropsin: float
    p_dropres: float
    p_droplyr: float
    p_nucleus: float
    c_beta: float
    c_gamma: float
    e_tie: bool
    e_preln: bool
    e_scale: str
    is_train: bool
    e_init: Initializer
    w_init: Initializer
    r_init: Initializer
    b_init: Initializer
    no_emb: bool = False

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in fields(TransformerConfig)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}

        if isinstance(filtered["param_dtype"], str):
            filtered["param_dtype"] = jnp.dtype(filtered["param_dtype"])

        if isinstance(filtered["dtype"], str):
            filtered["dtype"] = jnp.dtype(filtered["dtype"])

        for k, v in filtered.items():
            if signature[k] is bool and v in {0, 1}:
                filtered[k] = bool(v)

        filtered["e_init"] = inits.normal(1.0)
        filtered["w_init"] = inits.variance_scaling(1.0, "fan_in", "normal")
        filtered["r_init"] = inits.variance_scaling(1.0, "fan_in", "normal")
        filtered["b_init"] = inits.zeros

        return cls(**filtered)
