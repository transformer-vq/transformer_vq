import dataclasses
from typing import Any
from typing import Callable
from typing import List

import jax.nn.initializers as inits
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
    n_device: int
    global_batch_size: int
    sequence_len: int
    block_len: int
    attn_type: str
    head_type: str
    d_model: int
    d_k: int
    n_code: int
    n_layer: int
    n_vocab: int
    pe_abs: bool
    pe_lam: float
    p_dropsin: float
    p_dropres: float
    p_droplyr: float
    c_beta: float
    c_gamma: float
    e_tie: bool
    e_preln: bool
    e_scale: float
    is_train: bool
    e_init: Initializer
    w_init: Initializer
    b_init: Initializer
    no_emb: bool = False

    @classmethod
    def create(cls, **kwargs):
        signature = {f.name: f.type for f in dataclasses.fields(TransformerConfig)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}
        filtered["e_init"] = inits.normal(1.0)
        filtered["w_init"] = inits.variance_scaling(1.0, "fan_in", "normal")
        filtered["b_init"] = inits.zeros
        return cls(**filtered)
