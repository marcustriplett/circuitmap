import jax
from jax import jit
import jax.numpy as jnp
from functools import partial

@partial(jit, static_argnums=(0))
def get_mask(N):
    arr = jnp.ones((N, N))
    arr = jax.ops.index_update(arr, jnp.diag_indices(arr.shape[0]), 0)
    return arr.astype(bool)
