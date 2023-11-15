import jax.numpy as jnp
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.dataset = "enwik8"  # dataset name
    config.prng_seed = 42  # random seed
    config.global_batch_size = 8  # global batch size in sequences
    config.sequence_len = 800  # sequence length
    config.block_len = 50  # block length per cache update

    config.param_dtype = jnp.float32  # parameter dtype
    config.dtype = jnp.bfloat16  # activation dtype
    config.attn_type = "vq"  # one of vq, full, vq_old, full_old
    config.head_type = "shga"  # one of shga, mqa, mha
    config.d_model = 768  # model width
    config.d_k = 128  # key width
    config.n_code = 50  # number of codes per codebook
    config.n_layer = 2  # number of transformer layers (each is GAU+GAU or Attn+MLP)
    config.pe_abs = False  # include absolute positional embeddings?
    config.pe_lam = 100_000  # max angular wavelen for pos embs (similar to rope base)
    config.p_dropsin = 0.2  # dropout rate on rel position embs for xl biases
    config.p_dropres = 0.5  # dropout rate on residual connections
    config.p_droplyr = 0.3  # dropout rate for layerdrop
    config.c_beta = 0.0001  # vq commit loss coefficient
    config.c_gamma = 0.99  # vq codebook ema rate
    config.e_tie = False  # tie input/output embs?
    config.e_preln = False  # apply layernorm before projecting to vocab logits?
    config.e_scale = 1.0  # multiplier for output logits  (helps similar to z-loss)

    config.grad_clip = 0.1  # gradient clip (use None with adafactor)
    config.optimizer = "adamw"  # optimizer name: options are adamw, lion, adafactor
    config.lr_max = 0.0004  # peak learning rate
    config.lr_schedule = "cosine"  # learning rate decay schedule
    config.wd_lam = 0.0002  # decoupled weight decay coefficient
    config.n_warmup_step = 80  # steps of linear warmup
    config.n_max_step = 1000  # steps of training total
    config.n_print_step = 10  # steps of training per print
    config.n_save_step = 100  # steps of training per save
    config.n_eval_step = 10  # steps of evaluation per save
    config.n_save_keep = 5  # num checkpoints (save improvements, delete oldest)

    return config