import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import optax

from jaxtyping import Int, Array, PRNGKeyArray, Float, PyTree
from typing import Callable, Union
from optax import GradientTransformationExtraArgs

from tensorboardX import SummaryWriter



class ResBlock(eqx.Module):
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d
    act: Callable = eqx.field(static=True)

    def __init__(
            self,
            dim: Int,
            activation = jax.nn.relu,
            key: PRNGKeyArray | None = None
    ):
        key1, key2, key3 = jax.random.split(key, 3)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=(1,), key=key1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=(1,), key=key2)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=1, key=key3)

        self.act = activation
    
    def __call__(self, x: Array):
        y = x
        y = self.conv1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.act(y)
        y = self.conv3(y)

        y = y + x
        
        return y
    
class UpsampledConv(eqx.Module):
    conv: nn.Conv1d
    stride: int = eqx.field(static=True)
    
    def __init__(
            self,
            in_channels: Int,
            out_channels: Int,
            kernel_size: Union[Int, tuple[Int]],
            stride: Int,
            padding: Union[Int, str],
            key: PRNGKeyArray | None = None
    ):
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            key=key
        )
    
    def __call__(self, x: Array):
        upsampled_size = (x.shape[0], x.shape[1] * self.stride)
        upsampled = jax.image.resize(x, upsampled_size, method="nearest")
        return self.conv(upsampled)
    
class Encoder(eqx.Module):
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d
    res1: ResBlock
    res2: ResBlock
    res3: ResBlock

    def __init__(
            self,
            hidden_dim : Int = 1024,
            codebook_dim : Int = 512,
            key : PRNGKeyArray | None = None
    ):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.conv1 = nn.Conv1d(
            in_channels=80,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key1
        )

        self.conv2 = nn.Conv1d(
            in_channels=512,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key2
        )

        self.res1 = ResBlock(dim=hidden_dim, key=key3)
        self.res2 = ResBlock(dim=hidden_dim, key=key4)
        self.res3 = ResBlock(dim=hidden_dim, key=key5)

        self.conv3 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=codebook_dim,
            kernel_size=1,
            stride=1,
            key=key6
        )

    def __call__(
            self,
            x: Array
    ):
        y = self.conv1(x)
        y = jax.nn.relu(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.conv3(y)

        return y
    
class Decoder(eqx.Module):
    conv1: nn.Conv1d
    conv2: UpsampledConv
    conv3: UpsampledConv
    conv4: nn.Conv1d
    res1: ResBlock
    res2: ResBlock
    res3: ResBlock

    def __init__(
            self,
            hidden_dim : Int = 1024,
            codebook_dim : Int = 512,
            key : PRNGKeyArray | None = None
    ):
        key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)

        self.conv1 = nn.Conv1d(
            in_channels=codebook_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            key=key1
        )

        self.res1 = ResBlock(dim=hidden_dim, key=key2)
        self.res2 = ResBlock(dim=hidden_dim, key=key3)
        self.res3 = ResBlock(dim=hidden_dim, key=key4)

        self.conv2 = UpsampledConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key5
        )

        self.conv3 = UpsampledConv(
            in_channels=hidden_dim,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key6
        )

        self.conv4 = nn.Conv1d(
            in_channels=512,
            out_channels=80,
            kernel_size=1,
            stride=1,
            key=key7
        )

    def __call__(self, x: Array):
        y = self.conv1(x)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.conv3(y)
        y = jax.nn.relu(y)
        y = self.conv4(y)

        return y
    
class Quantizer(eqx.Module):
    K: Int = eqx.field(static=True)
    D: Int = eqx.field(static=True)
    codebook: jax.Array

    codebook_avg: jax.Array
    cluster_size: jax.Array

    decay: Float = eqx.field(static=True)
    eps: Float = eqx.field(static=True)

    def __init__(
            self,
            num_vecs: Int = 1024,
            num_dims: Int = 512,
            decay: Float = .99,
            eps: Float = 1e-5,
            key: PRNGKeyArray | None = None
    ):
        self.K = num_vecs
        self.D = num_dims

        self.decay = decay
        self.eps = eps

        self.codebook = jax.nn.initializers.variance_scaling(
            scale=1., mode="fan_in", distribution="uniform"
        )(key, (num_vecs, num_dims))

        self.codebook_avg = jnp.copy(self.codebook)
        self.cluster_size = jnp.zeros(num_vecs)

    def __call__(self, x: Array):
        flatten = jnp.reshape(x, (-1, self.D))
        a_squared = jnp.sum(flatten**2, axis=-1, keepdims=True)
        b_squared = jnp.transpose(jnp.sum(self.codebook**2, axis=-1, keepdims=True))

        distance = (a_squared + b_squared - 2 * jnp.matmul(flatten, jnp.transpose(self.codebook)))

        codebook_idx = jnp.argmin(distance, axis=-1)

        z_q = self.codebook[codebook_idx]

        z_q = flatten + jax.lax.stop_gradient(z_q - flatten)

        z_q = jnp.reshape(z_q, (-1, x.shape[-1]))

        return z_q, self.codebook_updates(flatten, codebook_idx)
    
    def codebook_updates(
            self,
            flatten: Array,
            codebook_idx: Array
    ):
        codebook_onehot = jax.nn.one_hot(codebook_idx, self.K)
        codebook_onehot_sum = jnp.sum(codebook_onehot, axis=0)
        codebook_sum = jnp.dot(flatten.T, codebook_onehot)

        new_cluser_size = (
            self.decay * self.cluster_size + (1 - self.decay) * codebook_onehot_sum
        )

        new_codebook_avg = (
            self.decay * self.codebook_avg + (1 - self.decay) * codebook_sum.T
        )

        n = jnp.sum(new_cluser_size)

        new_cluser_size = (new_cluser_size + self.eps) / (n + self.K * self.eps) * n
        new_codebook = self.codebook_avg / new_cluser_size[:,None]

        updates = (new_cluser_size, new_codebook_avg, new_codebook)

        return updates, codebook_idx
    
class VQVAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    quantizer: Quantizer

    def __init__(
            self,
            key: PRNGKeyArray
    ):
        key1, key2, key3 = jax.random.split(key, 3)

        self.encoder = Encoder(key=key1)
        self.decoder = Decoder(key=key2)
        self.quantizer = Quantizer(decay=.8, key=key3)

    def __call__(self, x: Array):
        z_e = self.encoder(x)
        z_q, codebook_indices = self.quantizer(z_e)
        y = self.decoder(z_q)

        return z_e, z_q, codebook_indices, y


def update_codebook_ema(model: VQVAE, updates: PyTree, codebook_idx: Array, key: PRNGKeyArray | None = None):
    avg_updates = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)
    h = jnp.histogram(
        codebook_idx, bins = model.quantizer.K, range=(0, model.quantizer.K)
    )[0] / len(codebook_idx)

    keep = 1 / model.quantizer.K

    mask = (h > 2 * keep) | (h < .5 * keep)

    rand_embed = (
        jax.random.normal(key, (model.quantizer.K, model.quantizer.D)) * mask[:,None]
    )

    avg_updates = (
        avg_updates[0],
        avg_updates[1],
        jnp.where(mask[:,None], rand_embed, avg_updates[2])
    )
    
    def where(q:VQVAE):
        return q.quantizer.cluster_size, q.quantizer.codebook_avg, q.quantizer.codebook
    
    model = eqx.tree_at(where, model, avg_updates)
    return model

@eqx.filter_value_and_grad(has_aux=True)
def calculate_losses_vqvae(model: Callable, x: Array):
    z_e, z_q, codebook_updates, y  = jax.vmap(model)(x)

    reconstruct_loss = jnp.mean(jnp.linalg.norm((x - y), ord=2, axis=(1,2)))

    commit_loss = jnp.mean(
        jnp.linalg.norm(z_e - jax.lax.stop_gradient(z_q), ord=2, axis=(1,2))
    )

    codebook = jnp.mean(codebook_updates[0][2], axis=0)

    KL_loss = .5 * jnp.sum(jnp.mean(codebook, axis=-1)**2 + jnp.var(codebook, axis=-1) - jnp.log(jnp.clip(jnp.std(codebook, axis=-1), min=1e-6)) - 1)

    total_loss = reconstruct_loss + commit_loss + KL_loss
    
    return total_loss, (reconstruct_loss, commit_loss, codebook_updates, y)

@eqx.filter_jit
def make_step(model: Callable, optimizer: GradientTransformationExtraArgs, opt_state: PyTree, x: Array, key: PRNGKeyArray):
    (total_loss, (reconstruct_loss, commit_loss, codebook_updates, y)), grads = (
        calculate_losses_vqvae(model, x)
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    model = update_codebook_ema(model, codebook_updates[0], codebook_updates[1], key)
    return (
        model,
        opt_state,
        total_loss,
        reconstruct_loss,
        commit_loss,
        codebook_updates,
        y,
    )

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    model = VQVAE(key)

    optim = optax.adam(1e-4)
    opt_state = optim.init(model)

    writer = SummaryWriter(log_dir="./runs")

    for i in range(100):
        key, subkey = jax.random.split(key)
        x = jnp.ones((10, 80, 100))
        model, opt_state, total_loss, reconstruct_loss, commit_loss, codebook_updates, y = (
            make_step(model, optim, opt_state, x, subkey)
        )
        print(f"Total loss: {total_loss}")
        writer.add_scalar("Loss/Total", total_loss, i)
        writer.add_scalar("Loss/Reconstruct", reconstruct_loss, i)
        writer.add_scalar("Loss/Commit", commit_loss, i)
        # writer.add_histogram('Codebook Updates/Cluster Size', jnp.mean(codebook_updates[0], axis=0), i)
        # writer.add_scalar('Codebook Updates/Codebook Avg', jnp.mean(codebook_updates[1], axis=0), i)
        writer.add_histogram(
            "Codebook Updates/Code ids used", jnp.sum(codebook_updates[1], axis=(0)), i
        )
        writer.add_histogram(
            "Codebook Updates/Code means", jnp.mean(codebook_updates[0][2], axis=(0, 2)), i
        )
        writer.add_histogram(
            "Codebook Updates/Code stds", jnp.std(codebook_updates[0][2], axis=(0, 2)), i
        )