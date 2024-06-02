from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax import random as jr
from jax import vmap
from jax.numpy import linalg as jla


def get_stationary(pi):
    mu = jla.svd(pi.T - jnp.eye(pi.shape[0]))[-1][-1]
    return mu / mu.sum()


class InContextTree:
    def __init__(self, vocab_size, dag, alpha):
        assert jnp.all(dag < jnp.arange(len(dag)))
        self.vocab_size = vocab_size
        self.dag = dag
        self.alpha = alpha

    def sample(self, key):
        pi_key, seq_key, test_key = jr.split(key, 3)
        prior = self.alpha * jnp.ones(self.vocab_size)
        pi = jr.dirichlet(pi_key, prior, [self.vocab_size])
        mu = get_stationary(pi)
        x = jnp.zeros((len(self.dag) + 1,), dtype=int)

        def step(i, carry):
            x, k = carry
            k, subkey = jr.split(k)
            p = jnp.where(self.dag[i] == -1, mu, pi[x[self.dag[i]]])
            x = x.at[i].set(jr.choice(subkey, pi.shape[0], p=p))
            return x, k

        x, _ = lax.fori_loop(0, len(self.dag), step, (x, seq_key))
        test_token = jr.choice(test_key, self.vocab_size)
        x = x.at[-1].set(test_token)
        y = pi[test_token]
        return x, y

    def bayes(self, seq):
        s, seq = seq[-1], seq[:-1]
        counts = jnp.zeros(self.vocab_size)
        counts = counts.at[seq].add(seq[self.dag] == s)
        counts += self.alpha
        return counts / counts.sum()


class InContextDAG:
    def __init__(self, vocab_size, dag, alpha):
        for i, p in enumerate(dag):
            print(i, p)
            assert max(p, default=-1) < i
        dag = [jnp.array(p, dtype=int) for p in dag]
        self.vocab_size = vocab_size
        self.dag = dag
        self.alpha = alpha

    def sample(self, key):
        pi_key, seq_key = jr.split(key)
        ks = set(len(p) for p in self.dag)
        pi_keys = jr.split(pi_key, len(ks))
        pi = dict()
        pi[0] = jnp.ones(self.vocab_size) / self.vocab_size
        prior = self.alpha * jnp.ones(self.vocab_size)
        for k, subkey in zip(ks, pi_keys):
            pi[k] = jr.dirichlet(subkey, prior, [self.vocab_size] * k)

        x = jnp.zeros((len(self.dag) - 1,), dtype=int)
        for i in range(len(self.dag)):
            k = len(self.dag[i])
            if k == 0:
                p = pi[0]
            else:
                p = pi[k][tuple(x[self.dag[i]])]

            if i != len(self.dag) - 1:
                seq_key, subkey = jr.split(seq_key)
                new_token = jr.choice(subkey, self.vocab_size, p=p)
                x = x.at[i].set(new_token)
        return x, p

    def bayes(self, seq):
        counts = jnp.zeros(self.vocab_size)
        s = seq[self.dag[-1]]
        for i in range(len(self.dag) - 1):
            if len(self.dag[i]) == len(s):
                counts = counts.at[seq[i]].add(jnp.all(seq[self.dag[i]] == s))
        counts += self.alpha
        return counts / counts.sum()
