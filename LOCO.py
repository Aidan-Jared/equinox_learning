import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Int, Array, PRNGKeyArray

from collections import deque
import torchvision
from sampler import MNIST_CL_loader

SEED = 42
KEY = jax.random.key(SEED)

class activationBuffer:
    def __init__(
            self,
            buffer_size: int,
            k: int
    ) -> None:
        self.buffer_size = buffer_size
        self.k = k

    def init(self):
        return []
    
    def addActivations(
            self,
            activations: list[Array],
            buffer_state: list,
            print_var: bool = False
    ) -> list[Array]:
        jit_pca = jax.jit(self._pca)
        for idx, i in enumerate(activations):
            A, explained_var = jit_pca(i)
            activations[idx] = A
            if print_var:
                print(explained_var)
        if buffer_state == []:
            buffer_state = activations
        else:
            for idx in range(len(buffer_state)):
                buffer_state[idx] = jnp.vstack((buffer_state[idx], activations[idx]))
                if buffer_state[idx].shape[-1] > 50:
                    buffer_state[idx] = jnp.delete(buffer_state[idx], 0)
        return buffer_state

    def _pca(
            self,
            x: Array
    ) -> tuple[Array, Float]:
        if x.ndim > 2:
            x = x.reshape(x.shape[0],-1)
        means = x.mean(axis=0, keepdims=True)
        x = x - means
        U, S, Vt = jax.scipy.linalg.svd(x, full_matrices=False)
        explained_var = (S[:self.k] ** 2) / (x.shape[0] - 1)
        A = Vt[:self.k] # needs work
        x = jnp.einsum("i ..., j ... -> j ...", x, A)
        # x = x @ A.T
        return x, explained_var

class LOCOLayer(eqx.Module):
    layer: eqx.Module

    def __init__(
            self,
            layer
    ):
        self.layer = layer
    
    def __call__(
            self,
            x: Array,
            keep_activations: bool = False,
            perterbation: Array | None = None
    ):
        x = self.layer(x)

        if perterbation is not None:
            x = x + perterbation

        if keep_activations:
            return x, x
        return x
         
class CNN(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4, key5 = jax.random.split(key,5)
        
        self.layers = [
            eqx.nn.Conv2d(1,15, kernel_size=4, key=key1),
            eqx.nn.Conv2d(15, 30, kernel_size=4, key=key5),
            # eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(22 * 22 * 30,512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512,64, key = key3),
            jax.nn.relu,
            eqx.nn.Linear(64,10, key = key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x
    
class LOCOTracker:

    def forward_pass(
            model: eqx.Module,
            x: Array,
            perterbation: Array| None = None,
            keep_activations: bool = False
    ):
        outputs = []
        current_x = x
        pert_idx = 0

        def apply_layer(layer, x_in):
            nonlocal pert_idx
            if isinstance(layer, LOCOLayer):
                if perterbation is not None:
                    pert = perterbation[pert_idx]
                    pert_idx += 1
                else:
                    pert = None
                
                if keep_activations:
                    out, stored = layer(x_in, keep_activations, pert)
                    outputs.append(stored)
                    return out
                
                else:
                    return layer(x_in, keep_activations, pert)
            else:
                out = layer(x_in) if callable(layer) else x_in
                return out
        
        if hasattr(model, "layers") and isinstance(model.layers, (list, tuple)):
            for layer in model.layers:
                current_x = apply_layer(layer, current_x)
        else:
            current_x = model(current_x)
        
        if keep_activations:
            return current_x, outputs
        return current_x
    
    @staticmethod
    def dual_forward(
        model: eqx.Module,
        x: Array,
        key: PRNGKeyArray,
        sigma = .01,
    ):
        output, activations = LOCOTracker.forward_pass(model, x, keep_activations = True)

        perturbations = [
            sigma * jax.random.normal(
                jax.random.fold_in(key, i), act.shape
            ) for i, act in enumerate(activations)
        ]

        output_pert = LOCOTracker.forward_pass(model, x, perturbation=perturbations, keep_activations = False)

        return {
            "output" : output,
            "output_pert" : output_pert,
            "activations" : activations,
        }


def wrap_for_loco(
        model: eqx.Module,
        layer_filter = lambda x : isinstance(x (eqx.nn.Linear, eqx.nn.Conv2d))
):
    def wrap_layer(x):
        if layer_filter(x):
            return LOCOLayer(x)
        return x
    
    return jax.tree.map(
        wrap_layer,
        model,
        is_leaf=layer_filter
    )

def update(model, TD_loss, eps, sigma, buffer_state, c, key, iter):
    updated_layers = []
    buffer_state = dict(zip(model.layer_idx, buffer_state))
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, eqx.Module) and idx in buffer_state:
            where = lambda m: m.weight
            
            updated_layer = eqx.tree_at(
                where,
                layer,
                replace_fn=lambda w: update_weights(
                    w, TD_loss, eps, sigma, buffer_state[idx], c, key, iter
                )
            )
            updated_layers.append(updated_layer)
        else:
            updated_layers.append(layer)
    
    return eqx.tree_at(lambda m: m.layers, model, updated_layers)

def kmeans(
        activation: Array,
        c: Int,
        key: PRNGKeyArray,
        iter: int
) -> Array:
    indicies = jax.random.choice(key, activation.shape[0], shape=(c,), replace=False)
    centroids = jnp.take(activation, indicies, axis=0)
    
    def compute_distance(a:Array, cent:Array):
        return jnp.sqrt(jnp.sum((a - centroids)**2, axis=-1))
    
    def update_centroid(i):
        mask = jnp.equal(assignment, i)
        masked_activations = jnp.where(mask, activation, 0)
        return jnp.sum(masked_activations, axis=0) / jnp.sum(mask)

    for _ in range(iter):
        distances = jax.vmap(compute_distance, in_axes=(0, None))(activation, centroids)
        assignment = jnp.argmin(distances, axis=-1)    
        centroids = jax.vmap(update_centroid)(jnp.arange(c))

    distances = jax.vmap(compute_distance, in_axes=(0, None))(activation, centroids)
    assignment = jnp.argmin(distances, axis=-1)

    mask = jnp.arange(c) != assignment

    return centroids[mask]

def P_CO(
        activation: Array,
        c: Int,
        key: PRNGKeyArray,
        iter: int
):
    A = kmeans(activation,  c, key, iter)
    I = jnp.eye(A.shape[0], dtype=A.dtype)
    return I - A @ jnp.linalg.inv(A.T @ A) @ A.T

def P_LOCO(
    activation: Array,
    c: Int,
    key: PRNGKeyArray,
    iter: int  
):
    P = P_CO(activation, c, key, iter)
    Z = P @ activation
    Q, _ = jnp.linalg.qr(Z)
    return Q @ jnp.linalg.inv(Q.T @ Q) @ Q.T

def update_weights(
    weight,
    TD_loss,
    eps,
    sigma,
    activation: Array,
    c: Int,
    key: PRNGKeyArray,
    iter: int 
):
    p_loco = P_LOCO(activation, c, key, iter)
    d_weight = - sigma * TD_loss * jnp.outer(eps, (p_loco @ activation[-1]))
    return weight + d_weight


if __name__ == "__main__":
    normalize_data = torchvision.transforms.Compose(
        [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset= torchvision.datasets.MNIST(
        "MNIST",
        train= True,
        download=True,
        transform=normalize_data,
    )

    test_dataset= torchvision.datasets.MNIST(
        "MNIST",
        train= False,
        download=True,
        transform=normalize_data,
    )

    subkey1, subkey2, subkey3 = jax.random.split(KEY, 3)
    trainloader = MNIST_CL_loader(train_dataset, subkey1, 32, 2)
    testloader = MNIST_CL_loader(test_dataset, subkey2, 2, 2)

    model = LOCOCNN(subkey3)
    buffer = activationBuffer(10, 1)
    buffer_state = buffer.init()

    x, y = trainloader.sample(0, "cpu")

    y, activations = jax.vmap(model, in_axes=(0, None))(x, subkey3)

    buffer_state = buffer.addActivations(activations, buffer_state)

    update(model, 1.0, 3, 2, buffer_state, 1, subkey1, 10)

    print("hi")