import jax
import jax.numpy as jnp

from jax import vmap, jit
from jax.experimental import loops
from jax.ops import index_update

@jit
def simultaneous_isotonic_regression(X, Ys, y_min=0, y_max=1):
    """
    Run PAVA simultaneously on many problem instances. 
    Each problem must have the same independent variables, but we parallelize
    over different observations.
    
    Args
        X: (num_measurements)
        Ys: (num_problems x num_measurements)
        
    Returns
    
        Y_hats: (num_problems x num_measurements) where each row is the result of running an isotonic
                 regression.
    
    """
    
    
    idx = jnp.argsort(X)
    X_s = X[idx]
    Y_s = Ys[:,idx]
    
    Y_preds = vmap(_isotonic_regression, in_axes=(0, None))(Y_s, jnp.ones_like(Y_s[0,:]))
    return jnp.clip(Y_preds, a_min=y_min, a_max=y_max)

@jit
def _isotonic_regression(y, weight):

    def true_fun(args):
        (i, k, solution, numerator, denominator, pooled) = args
        with loops.Scope() as s:
            s.i = i
            s.k = k
            s.solution = solution
            s.numerator = 0.0
            s.denominator = 0.0
            s.pooled = pooled
            
            s.j = s.i
            for _ in s.while_range(lambda: s.j < s.k + 1):
                s.numerator += s.solution[s.j] * weight[s.j]
                s.denominator += weight[s.j]
                s.j += 1
                
            s.j = s.i
            for _ in s.while_range(lambda: s.j < s.k + 1):
                s.solution = index_update(s.solution, s.j, s.numerator / s.denominator)
                s.j += 1
            s.pooled = 1
            return s.solution, s.numerator, s.denominator, s.pooled

    def false_fun(args):
        return s.solution, s.numerator, s.denominator, s.pooled
        
    weight = jnp.array(weight, copy=True)
    y = jnp.array(y, copy=True)
    n = y.shape[0]
    with loops.Scope() as s:
        s.exit_early = (n <= 1)
        s.n = n - 1
        s.pooled = 1
        s.i = 0
        s.solution = jnp.array(y, copy=True, dtype=float)
        s.k = 0
        s.numerator = 0.0
        s.denominator = 0.0
        

        for _ in s.while_range(lambda:
            jnp.logical_and(s.pooled > 0, jnp.logical_not(s.exit_early))):
            s.exit_early = False
            s.i = 0
            s.pooled = 0
            
            for _ in s.while_range(lambda: s.i < s.n):
                s.k = s.i
                for _ in s.while_range(
                    lambda: jnp.logical_and(s.k < s.n, s.solution[s.k] >= s.solution[s.k + 1])
                ):
                    s.k += 1
                args = (s.i, s.k, s.solution, s.numerator, s.denominator, s.pooled)
                s.solution, s.numerator, s.denominator, s.pooled = jax.lax.cond(
                    s.solution[s.i] != s.solution[s.k],
                    true_fun,
                    false_fun,
                    args
                )
                
                s.i = s.k + 1

        return s.solution