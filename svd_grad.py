import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PyTree

class SVD_grad:
    def __init__(
            self,
            optim: optax.GradientTransformationExtraArgs,
            threshold: Float = 0.1
            ) -> None:
        self.optim = optim
        self.threshold = threshold
        self.stored_subspace = None
        self.use_projection = False
    
    def _get_projected_subspace(
            self,
            params: PyTree
    ):
        def _project(w):
            # if len(w.shape) < 2:
            #     w = jnp.expand_dims(w,axis=1)
            w = w.reshape(w.shape[0],-1)
            U, S, Vh = jnp.linalg.svd(w)
            threshold = jnp.max(S) * self.threshold
            k = jnp.sum(S > threshold)

            if k == 0:
                return jnp.zeros((w.shape[0], 1))

            projected_subspace = U[:, :k]
            return projected_subspace
        
        
        return jax.tree.map(
            _project,
            params
        )
    
    def _project_grads(
            self,
            grads: PyTree,
            params: PyTree
    ):
        
        if not self.use_projection or self.stored_subspace is None:
            return grads
        
        # A = self._get_projected_subspace(params)

        def project(g, a):
            a_flat = a.reshape(a.shape[0], -1)
            g_flat = g.reshape(g.shape[0], -1)
            g_shape = g.shape

            p = jnp.identity(g_shape[0]) - a_flat @ a_flat.T

            grads_projected = p @ g_flat

            return grads_projected.reshape(g_shape)

        return jax.tree.map(
            project,
            grads, self.stored_subspace
        )  
    
    def init(
            self,
            Params: PyTree
    ):
        return self.optim.init(Params)
    
    def update(
            self,
            grads: PyTree,
            state: PyTree,
            params: PyTree
    ):
        
        grads_projected = self._project_grads(grads, params)
        
        return self.optim.update(grads_projected, state, params)
    
    def start_new_task(
            self,
            params: PyTree
    ):
        self.stored_subspace = self._get_projected_subspace(params)
        self.use_projection = True
    
    def disable_projection(self):
        self.use_projection = False

    def enable_projection(self):
        self.use_projection = True
    
    def clear_subspace(self):
        self.stored_subspace = None