import dataclasses

import flax.linen as nn
import jax.numpy as jnp

from transformer_vq.nn.types import TransformerConfig


class Embeddings(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.apply_config()
        emb_args = [self.e_init, [self.n_vocab, self.d_model], self.param_dtype]
        self.embs = self.param("embs", *emb_args)
        bias_out_args = [self.b_init, [self.n_vocab], self.param_dtype]
        self.bias_out = self.param("bias_out", *bias_out_args)

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def __call__(self, x):
        x = jnp.take_along_axis(
            self.embs[None, ...], x[..., None].astype(jnp.int32), axis=1
        )
        return x.astype(self.dtype)

    def logits(self, x):
        x = x.astype(jnp.float32)
        x = jnp.dot(x, self.embs.T.astype(jnp.float32))
        x += self.bias_out.astype(jnp.float32)[None, None, ...]
        return x
