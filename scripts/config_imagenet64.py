import jax.numpy as jnp
from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.dataset = "imagenet64"  # dataset name
    config.prng_seed = 42  # random seed
    config.global_batch_size = 16  # global batch size in sequences
    config.sequence_len = 12288  # sequence length
    config.block_len = 512  # block length per cache update

    config.param_dtype = jnp.float32  # parameter dtype
    config.dtype = jnp.bfloat16  # activation dtype
    config.attn_type = "vq"  # one of vq, full, vq_old, full_old
    config.head_type = "shga"  # one of shga, mqa, mha
    config.d_model = 2048  # model width
    config.d_k = 128  # key width
    config.n_code = 512  # number of codes per codebook
    config.n_layer = 24  # number of transformer layers (each is GAU+GAU or Attn+MLP)
    config.pe_abs = False  # include absolute positional embeddings?
    config.pe_lam = 100_000  # max angular wavelen for pos embs (similar to rope base)
    config.p_dropsin = 0.1  # dropout rate on rel position embs for xl biases
    config.p_dropres = 0.0  # dropout rate on residual connections
    config.p_droplyr = 0.0  # dropout rate for layerdrop
    config.c_beta = 0.0001  # vq commit loss coefficient
    config.c_gamma = 0.99  # vq codebook ema rate
    config.e_tie = True  # tie input/output embs?
    config.e_preln = True  # apply layernorm before projecting to vocab logits?
    config.e_scale = 0.005  # multiplier for output logits  (helps similar to z-loss)

    config.grad_clip = None  # gradient clip (use None with adafactor)
    config.optimizer = "adafactor"  # optimizer name: options are adamw, lion, adafactor
    config.lr_max = 0.01  # peak learning rate
    config.lr_schedule = "cosine"  # learning rate decay schedule
    config.wd_lam = 0.0  # decoupled weight decay coefficient
    config.n_warmup_step = 10_000  # steps of linear warmup
    config.n_max_step = 500_000  # steps of training total
    config.n_print_step = 100  # steps of training per print
    config.n_save_step = 1_000  # steps of training per save
    config.n_eval_step = 100  # steps of evaluation per save
    config.n_save_keep = 5  # num checkpoints (save improvements, delete oldest)

    return config
