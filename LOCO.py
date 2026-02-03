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
            # key = jax.random.key(SEED)
            # for _ in range(self.buffer_size-1):
            #     subkey, key = jax.random.split(key)
            #     buffer_state.append([jax.random.normal(jax.random.fold_in(subkey, i), shape=layer.shape) for i, layer in enumerate(activations)])
            buffer_state = activations
        else:
            # [buffer[layer_idx] for buffer in buffer_state]
            buffer_state = [jnp.vstack((j, activations[idx])) for idx, j in enumerate(buffer_state)]
            if buffer_state[0].shape[0] > self.buffer_size:
                buffer_state = [jnp.delete(j, 0, assume_unique_indices = True, axis=0) for j in buffer_state]
        return buffer_state

    def _pca(
            self,
            x: Array
    ) -> tuple[Array, Float]:
        if x.ndim > 2:
            x = x.mean(2)
            x = x.reshape(x.shape[0],-1)
        means = x.mean(axis=0, keepdims=True)
        x = x - means
        U, S, Vt = jax.scipy.linalg.svd(x, full_matrices=False)
        explained_var = (S[:self.k] ** 2) / (x.shape[0] - 1)
        A = Vt[:self.k] # needs work
        x = jnp.einsum("i ..., j ... -> j ...", x, A)
        # x = x @ A.T
        return jnp.expand_dims(x,0), explained_var

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
            key: PRNGKeyArray | None = None,
            eps: Float | None = None
    ):
        out = self.layer(x)

        if key is not None:
            perterbation = jax.random.normal(key, out.shape) * eps
            out = out + perterbation

        if keep_activations:
            # if hasattr(self.layer,"kernel_size"):
            #     activation = jax.lax.conv_general_dilated_patches(
            #         lhs = jnp.expand_dims(x,0), filter_shape = self.layer.kernel_size,
            #         window_strides= self.layer.stride,
            #         padding= self.layer.padding
            #     )
            # else:
            activation = x
            return out, activation
        return out
         
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

    def __call__(self, x: Float[Array, "1 28 28"], key: PRNGKeyArray |None = None) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x, key= key)
        return x
    
class LOCOTracker:

    def forward_pass(
            self,
            model: eqx.Module,
            x: Array,
            key: PRNGKeyArray| None = None,
            sigma: Float | None = None,
            keep_activations: bool = False
    ):
        outputs = []
        current_x = x

        def apply_layer(layer, x_in, key):
            if isinstance(layer, LOCOLayer):
                # if key is not None:
                #     pert = key[pert_idx]
                #     pert_idx += 1
                # else:
                #     pert = None
                
                if keep_activations:
                    out, stored = layer(x_in, keep_activations, key, sigma)
                    outputs.append(stored)
                    return out
                
                else:
                    return layer(x_in, keep_activations, key, sigma)
            else:
                out = layer(x_in) if callable(layer) else x_in
                return out
        
        if hasattr(model, "layers") and isinstance(model.layers, (list, tuple)):
            for layer in model.layers:
                if key is not None:
                    key, subkey = jax.random.split(key)
                else:
                    subkey = None
                current_x = apply_layer(layer, current_x, subkey)
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
        output, activations = LOCOTracker().forward_pass(model=model, x=x, keep_activations = True)

        # perturbations = [
        #     sigma * jax.random.normal(
        #         jax.random.fold_in(key, i), act.shape
        #     ) for i, act in enumerate(activations)
        # ]

        output_pert = LOCOTracker().forward_pass(model, x, key=key, keep_activations = False, sigma = sigma)

        return {
            "output" : output,
            "output_pert" : output_pert,
            "activations" : activations,
        }


def wrap_for_loco(
        model: eqx.Module,
        layer_filter = lambda x : isinstance(x, (eqx.nn.Linear, eqx.nn.Conv2d))
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

def TD_loss(
        results: dict,
        y: Array
):
    def cross_entropy(
        y: Int[Array, " batch"],
        pred_y: Float[Array, "batch 10"]
    ) -> Float[Array, ""]:
        pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y,1), axis=1)
        return -jnp.mean(pred_y).item()

    return cross_entropy(y, results["output"]) - cross_entropy(y, results["output_pert"])

def update(model, TD_loss, eps, sigma, buffer_state, c, key, iter):
    updated_layers = []
    layer_idx = 0
    for layer in model.layers:
        if isinstance(layer, LOCOLayer):
            # layer_buffer = [buffer[layer_idx] for buffer in buffer_state]
            where = lambda m: m.layer
            
            updated_layer = eqx.tree_at(
                where,
                layer,
                replace_fn=lambda w: update_weights(
                    w, TD_loss, eps, sigma, buffer_state[layer_idx], c, key, iter
                )
            )
            layer_idx += 1
            updated_layers.append(updated_layer)
        else:
            updated_layers.append(layer)
    
    return eqx.tree_at(lambda m: m.layers, model, updated_layers)

def kmeans(
        activation: Array,
        seen_tasks: Int,
        key: PRNGKeyArray,
        iter: int
) -> Array:
    indicies = jax.random.choice(key, activation.shape[0], shape=(seen_tasks,), replace=False)
    centroids = jnp.take(activation, indicies, axis=0)
    
    def compute_distance(a:Array, cent:Array):
        return jnp.sqrt(jnp.einsum("cpl -> c", (a-cent)**2))
    
    def update_centroid(i):
        mask = jnp.expand_dims(jnp.equal(assignment, i), axis=1)
        masked_activations = jnp.where(mask, activation, 0)
        new_centroid = jnp.sum(masked_activations, axis=1) / jnp.sum(mask)
        new_centroid = jax.lax.cond(
            jnp.isnan(new_centroid).any(),
            lambda n: jnp.zeros_like(n),
            lambda n: n,
            new_centroid
        )
        # if jnp.isnan(new_centroid).any():
        #     new_centroid = jnp.zeros_like(new_centroid)
        return new_centroid

    for _ in range(iter):

        distances = jax.vmap(compute_distance, in_axes=(0, None))(activation, centroids)
        assignment = jnp.argmin(distances, axis=-1)
        # update_centroid(0)
        centroids = jax.vmap(update_centroid)(jnp.arange(seen_tasks))

    distances = compute_distance(activation[-1], centroids)
    assignment = jnp.argmin(distances, axis=-1)

    mask = jnp.arange(seen_tasks) != assignment

    return centroids[mask]

def P_CO(
        activation: Array,
        c: Int,
        key: PRNGKeyArray,
        iter: int
):
    A = kmeans(activation,  c, key, iter).squeeze(0)
    I = jnp.eye(A.shape[0], dtype=A.dtype)
    return jnp.nan_to_num(I - A @ jnp.linalg.inv(A.T @ A) @ A.T)

def P_LOCO(
    activation: Array,
    c: Int,
    key: PRNGKeyArray,
    iter: int  
):
    P = P_CO(activation, c, key, iter)
    Z = P @ activation 
    Q, _ = jnp.linalg.qr(Z)
    return jnp.nan_to_num(Q @ jnp.linalg.inv(Q.T @ Q) @ Q.T)

def update_weights(
    layer,
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
    if layer.bias:
        layer.bias = layer.bias + d_weight
    layer.weight = layer.weight + d_weight
    return layer


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

    subkey1, subkey2, subkey3, subkey4 = jax.random.split(KEY, 4)
    trainloader = MNIST_CL_loader(train_dataset, subkey1, 32, 2)
    testloader = MNIST_CL_loader(test_dataset, subkey2, 2, 2)

    model = CNN(subkey3)
    model = wrap_for_loco(model)
    buffer = activationBuffer(10, k = 10)
    buffer_state = buffer.init()
    trainer = LOCOTracker()

    for _ in range(10):
        x, y = trainloader.sample(0, "cpu")
        _,activations = jax.vmap(trainer.forward_pass, in_axes=(None, 0, None, None, None))(model, x, None, None, True)
        buffer_state = buffer.addActivations(activations, buffer_state)
    x, y = trainloader.sample(0, "cpu")

    results = jax.vmap(trainer.dual_forward, in_axes=(None, 0, None))(model, x, subkey4)
    td_loss = TD_loss(results, y)
    activations = results["activations"]

    # y, activations = jax.vmap(model, in_axes=(0, None))(x, subkey3)

    buffer_state = buffer.addActivations(activations, buffer_state)

    update(model, TD_loss=td_loss, eps=3, sigma=2, buffer_state=buffer_state, c=2, key=subkey1, iter=10)

    print("hi")