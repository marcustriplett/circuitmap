''' @author Benjamin Antin, Columbia University
'''

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

def _make_pava_pools(y, gamma=1.0):
	y = jnp.array(y)
	lg = jnp.log(gamma)
	T = len(y)

	v = jnp.zeros_like(y)
	w = jnp.zeros_like(y)
	l = jnp.zeros_like(y, dtype=jnp.int32)

	v = v.at[0].set(y[0])
	w = w.at[0].set(1)
	l = l.at[0].set(1)

	i = 0  # index of last pool

	def _outer_loop_fun(t, outer_state):
		(i, v, w, l) = outer_state
		i += 1

		v = v.at[i].set(y[t])
		w = w.at[i].set(1)
		l = l.at[i].set(1)

		(i, t, v, w, l) = jax.lax.while_loop(
			_check_inner_loop,
			_inner_loop_fun,
			(i, t, v, w, l)
		)
		return (i, v, w, l)

	def _check_inner_loop(state):
		(i, t, v, w, l) = state
		return jnp.all(jnp.array([i > 0,
			(v[i-1] / w[i-1] * jnp.exp(lg*l[i-1]) > (v[i] / w[i]))
		]))
	
	def _inner_loop_fun(state):
		(i, t, v, w, l) = state
		i -= 1

		v = v.at[i].set(v[i] + v[i+1] * jnp.exp(lg*l[i]))
		w = w.at[i].set(w[i] + w[i+1] * jnp.exp(2*lg*l[i]))
		l = l.at[i].set(l[i] + l[i+1])
		
		l = l.at[i+1].set(0)
		w = w.at[i+1].set(0)
		v = v.at[i+1].set(0)
		return (i, t, v, w, l)

	outer_state = (i, v, w, l)
	state = jax.lax.fori_loop(1, T, _outer_loop_fun, outer_state)
	(i, v, w, l) = state
	return v, w, l

def _reconstruct_from_pools(y_orig, v, w, l, gamma):
	out = jnp.zeros_like(y_orig)
	lg = jnp.log(gamma)

	def _add_one_pool(i, state):
		t_curr, out = state
		value, weight, length = v[i], w[i], l[i]
		out = jax.lax.fori_loop(
			t_curr.astype(int), (t_curr + length).astype(int),
			lambda j, out: out.at[j].set(value / weight * jnp.exp(lg * (j - t_curr))),
			out)
		t_curr += length
		return (t_curr, out)

	state = (0, out)
	state = jax.lax.fori_loop(0, len(v), _add_one_pool, state)
	(t_curr, out) = state
	return out

# def pava_decreasing(y, gamma=1.0):
# 	v, w, l = _make_pava_pools(-y, gamma=gamma)
# 	return -_reconstruct_from_pools(y, v, w, l, gamma)

def _isotonic_regression(y, gamma=1.0):
	v, w, l = _make_pava_pools(y, gamma=gamma)
	return _reconstruct_from_pools(y, v, w, l, gamma)
