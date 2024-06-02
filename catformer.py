import jax
from jax import nn
from jax import numpy as jnp
from jax import random as jr
from simple_pytree import Pytree, static_field


class muLinear(Pytree):
    def __init__(self, input_dim, output_dim, key, bias=True, zero_init=False):
        if zero_init:
            self.W = jnp.zeros([input_dim, output_dim])
        else:
            self.W = jr.normal(key, [input_dim, output_dim]) / jnp.sqrt(output_dim)
        if bias:
            self.b = jnp.zeros(output_dim)

    def __call__(self, x):
        x @= self.W * jnp.sqrt(self.W.shape[1] / self.W.shape[0])
        if hasattr(self, "b"):
            x += self.b * jnp.sqrt(len(self.b))
        return x


class MLP(Pytree):
    activation = static_field()

    def __init__(self, widths, activation, key):
        self.activation = activation

        keys = jr.split(key, len(widths) - 1)
        layers = []
        for i in range(len(widths) - 2):
            layers.append(muLinear(widths[i], widths[i + 1], keys[i]))
        layers.append(muLinear(widths[-2], widths[-1], keys[-1], zero_init=True))
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class CatFormer(Pytree):
    vocab_size: int = static_field()

    def attn(self, x, A):
        T = x.shape[-2]
        attn = jnp.einsum("...ij,jk,...lk -> ...il", x, A, x)
        attn = jnp.where(jnp.tri(T), attn, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x)
        return attn

    def embed(self, x):
        wte = jnp.eye(self.vocab_size)[x]
        wpe = jnp.eye(x.shape[-1])
        wpe = jnp.broadcast_to(wpe, (*x.shape, x.shape[-1]))
        return jnp.concatenate([wte, wpe], -1)

    def __init__(
        self,
        seq_len,
        vocab_size,
        heads,
    ):
        self.vocab_size = vocab_size
        d = seq_len + vocab_size
        self.A = []
        for n_head in heads:
            self.A.append(jnp.zeros([n_head, d, d]))
            d *= 1 + n_head
        self.W = jnp.zeros((d, vocab_size))

    def __call__(self, x):
        x = self.embed(x)
        for Ai in self.A:
            attn = jax.vmap(self.attn, (None, 0), -2)(x, Ai)
            attn = attn.reshape(*attn.shape[:-2], -1)
            x = jnp.concatenate([x, attn], -1)
        x = x[..., -1, :]
        return nn.softmax(x @ self.W)
    
class Transformer(Pytree):
    vocab_size: int = static_field()

    def attn(self, x, A):
        T = x.shape[-2]
        attn = jnp.einsum("...ij,jk,...lk -> ...il", x, A, x)
        attn = jnp.where(jnp.tri(T), attn, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x)
        return attn

    def embed(self, x):
        wte = self.wte[x]
        wpe = jnp.broadcast_to(self.wpe, (*x.shape, self.wpe.shape[-1]))
        return wte + wpe
        
    def __init__(
        self,
        seq_len,
        vocab_size,
        heads,
        key
    ):
        
        self.vocab_size = vocab_size
        d = seq_len + vocab_size
        self.A = []
        self.V = []
        self.mlps = []
        keys = jr.split(key, 2*len(heads)+2)
        for i in range(len(heads)):
            n_head = heads[i]
            self.A.append(jnp.zeros([n_head, d, d]))
            self.V.append(jr.normal(keys[2*i],[d, d])/jnp.sqrt(d))
            self.mlps.append(MLP(widths = [d, d, d], activation = nn.relu, key = keys[2*i + 1]))
        self.W = jnp.zeros((d, vocab_size))
        
        
        self.wte = jr.normal(keys[-2], [vocab_size, d]) / jnp.sqrt(d)
        self.wpe = jr.normal(keys[-1], [seq_len, d]) / jnp.sqrt(d)

        
    def get_attn(x):
        x = self.embed(x)

    def __call__(self, x):
        x = self.embed(x)

        for i in range(len(self.A)):
            Ai = self.A[i]
            Vi = self.V[i]
            mlpi = self.mlps[i]
            attn = jax.vmap(self.attn, (None, 0), -2)(x, Ai)
            attn = attn.reshape(*attn.shape[:-2], -1)
            delta = attn@Vi
            delta = jax.vmap(mlpi)(delta)
            x = x + delta
        x = x[..., -1, :]
        return nn.softmax(x @ self.W)
