import jax
from jax import numpy as jnp, vmap, jit, lax, random as jr, tree_util as jtu
from jax.numpy import linalg as jla
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node
from flax import linen as nn
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from tqdm.auto import tqdm, trange
from collections import namedtuple
from functools import partial
from jaxopt.tree_util import *
from typing import (
    Any,
    Callable,
    Union,
    List,   
)

class RNG:
    def __init__(self, seed=None, key=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        elif key is not None:
            self.key = key
        else:
            raise Exception("RNG expects either a seed or random key.")

    def next(self, n_keys=1):
        if n_keys > 1:
            return jax.random.split(self.next(), n_keys)
        else:
            self.key, key = jax.random.split(self.key)
            return key

    def __getattr__(self, name):
        return partial(getattr(jax.random, name), self.next())


register_pytree_node(
    RNG,
    lambda rng: ((rng.key,), None),
    lambda _, c: RNG(key=c[0]),
)