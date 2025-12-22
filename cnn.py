import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torchvision
from jaxtyping import Array, Float, Int, PyTree
from svd_grad import SVD_grad
from sampler import MNIST_CL_loader


BATCH_SIZE = 32
LEARNING_RATE = 1E-3
STEPS = 150
PRINT_EVERY = 150
SEED = 42

KEY = jax.random.PRNGKey(SEED)

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

def cross_entropy(
        y: Int[Array, " batch"],
        pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y,1), axis=1)
    return -jnp.mean(pred_y)
    

def loss( model: CNN, x: Float[Array, " batch 1 28 28"], y: Int[Array, " batch"]) -> Float[Array, ""]:
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)

def loss2(params, static, x, y):
    model = eqx.combine(params, static)
    return loss(model, x, y)

# @eqx.filter_jit
def compute_accuracy(
    model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)

def evaluate(model: CNN, testloader: MNIST_CL_loader, task: Int, steps):
    avg_loss = 0
    avg_acc = 0
    n = BATCH_SIZE * steps
    for _ in range(steps):
        x, y = testloader.sample(task)
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss/steps, avg_acc/steps

def train(
        model: CNN,
        trainloader: MNIST_CL_loader,
        testloader: MNIST_CL_loader,
        optim: optax.GradientTransformation | SVD_grad,
        steps_per_task: int,
        tasks: Int,
        print_every: int
) -> CNN:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # @eqx.filter_jit
    def make_step(
        model: CNN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    seen_tasks = []
    for i in range(tasks):
        seen_tasks.append(i)
        for step in range(steps_per_task):
            x, y = trainloader.sample(i)
            model, opt_state, train_loss = make_step(model, opt_state, x, y)
            if (step % print_every) == 0 or (step == steps_per_task - 1):
                for j in seen_tasks:
                    test_loss, test_accuracy = evaluate(model, testloader, j, steps_per_task // 10)
                    print(
                        f"task {j}, train_loss={train_loss.item()}, "
                        f"test_loss={test_loss}, test_accuracy={test_accuracy}"
                    )
        optim.start_new_task(eqx.filter(model, eqx.is_array))
    return model

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

    # trainloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=BATCH_SIZE, shuffle=True
    # )

    # testloader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=BATCH_SIZE, shuffle=True
    # )

    subkey1, subkey2, subkey3 = jax.random.split(KEY, 3)
    trainloader = MNIST_CL_loader(train_dataset, subkey1, BATCH_SIZE, 2)
    testloader = MNIST_CL_loader(test_dataset, subkey2, BATCH_SIZE, 2)


    
    model = CNN(subkey3)

    params, static = eqx.partition(model, eqx.is_array)

    optim = optax.adamw(LEARNING_RATE)
    optim = SVD_grad(optim, threshold=.8)

    model = train(model, trainloader, testloader, optim, STEPS, 5, PRINT_EVERY)
