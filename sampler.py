import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset
from jaxtyping import PRNGKeyArray, Int, Array, Float


# add batching (groups of tasks, 1 batch is 1 task)
class MNIST_CL_loader:
    def __init__(
            self,
            dataset: Dataset, 
            key: PRNGKeyArray, 
            batch_size: Int,
            class_p_task: Int = 5,
            device: str = "cpu"
            ) -> None:
        
        self.key=key
        self.class_p_task=class_p_task

        self.batch_size = batch_size
        self.seen_tasks = []

        self.len = len(dataset)

        class_to_indices = {}
        all_data = []

        for idx, (data, label) in enumerate(dataset):
            if isinstance(data, jnp.ndarray):
                all_data.append(np.array(data))
            else:
                all_data.append(data.numpy())
            label_int = int(label)
            if label_int not in class_to_indices:
                class_to_indices[label_int] = []
            class_to_indices[label_int].append(idx)
        
        device = jax.devices(device)[0]

        all_data_np = np.stack(all_data)
        # if all_data_np.ndim == 4 and all_data_np.shape[1] == 1:
        #     all_data_np = all_data_np.squeeze(1)

        self.all_data = jax.device_put(all_data_np, device)
        

        self.num_classes = len(class_to_indices)

        max_samples_per_class = max(len(v) for v in class_to_indices.values())

        self.class_indicies = jax.device_put(
            jnp.full((self.num_classes, max_samples_per_class), -1, dtype=jnp.int32), device
        )

        self.class_lenghts = jax.device_put(
            jnp.zeros(self.num_classes, dtype=jnp.int32), device
        )

        for class_idx, (label, idx) in enumerate(sorted(class_to_indices.items())):
            num_samples = len(idx)
            self.class_indicies = self.class_indicies.at[class_idx, :num_samples].set(
                jnp.array(idx, dtype=jnp.int32)
            )

            self.class_lenghts = self.class_lenghts.at[class_idx].set(num_samples)
        

        self.tasks = jnp.arange(self.num_classes).reshape((-1, self.class_p_task))
        
        self._sample_task_jit = jax.jit(
            self._sample_task_fn,
            static_argnames=["task_n", "batch_size", "class_p_task"]
        )

        # self._sample_batch_jit = jax.jit(
        #     jax.vmap(self._sample_task_fn, in_axes=(0, None, None, None, None, None, None)),
        #     static_argnames=["task", "batch_size",  "class_p_task"]
        # )

    def __len__(self):
        return self.len

    @staticmethod
    def _sample_task_fn(
        key: PRNGKeyArray,
        class_indicies: Array,
        all_data: Array,
        task_n: Int,
        batch_size: Int,
        tasks: Array,
        class_p_task: Int
    ) -> tuple[Array, Array]:
        
        # key, subkey = jax.random.split(key)
        # task = class_indicies[task_n]

        # shuffled_classes = jax.random.permutation(subkey, num_classes)
        # selected_classes = jax.lax.dynamic_slice(shuffled_classes, (0,),(n_ways,))
        
        def sample_class(
                carry: tuple[PRNGKeyArray, Int], 
                class_idx: Int
                ) -> tuple[tuple[PRNGKeyArray, Int], tuple[Array, Array]]:
            key, idx = carry
            key, subkey = jax.random.split(key)

            class_row = class_indicies[idx]
            mask = class_row > 0
            valid_idx = class_row[jnp.where(mask, class_row, 0)]

            selected_idx = jax.random.choice(subkey, valid_idx, shape=(batch_size // class_p_task,), replace=False)

            data = all_data[selected_idx]

            labels = jnp.full(batch_size // class_p_task, tasks[task_n][idx], dtype=jnp.int32)

            return (key, idx + 1), (data, labels)
        
        (key, _), (all_class_data, all_class_labels) = jax.lax.scan(
            sample_class,
            (key, 0),
            length=len(tasks[task_n])
        )

        all_class_data = all_class_data.reshape(-1, *all_class_data.shape[2:])
        all_class_labels = all_class_labels.reshape(-1)

        return all_class_data, all_class_labels
        
    def sample(
            self,
            task_n: Int,
            device: str = "gpu",
    ) -> tuple[Array, Array]:
        
        self.key, subkey = jax.random.split(self.key)
        
        all_data, all_labels = self._sample_task_jit(
            subkey,
            self.class_indicies,
            self.all_data,
            task_n,
            self.batch_size,
            self.tasks,
            self.class_p_task
        )
        if device == "gpu":
            device = jax.devices('gpu')[0]
            all_data = jax.device_put(all_data, device)
            all_labels = jax.device_put(all_labels, device)

        self.key, subkey = jax.random.split(self.key)

        shuffle_idx = jax.random.permutation(subkey, all_data.shape[0])
        all_data = all_data[shuffle_idx]
        all_labels = all_labels[shuffle_idx]
        
        # data_reshaped = all_data.reshape(self.n_ways, self.samples_per_class, *all_data.shape[1:])
        # labels_reshaped = all_labels.reshape(self.n_ways, self.samples_per_class)

        # support_data = data_reshaped[:,:self.k_shot].reshape(-1, *all_data.shape[1:])
        # query_data = data_reshaped[:,self.k_shot:].reshape(-1, *all_data.shape[1:])

        # support_labels = labels_reshaped[:, :self.k_shot].reshape(-1)
        # query_labels = labels_reshaped[:, self.k_shot:].reshape(-1)

        return all_data , all_labels
        
    
    def get_memory_usage(self) -> dict:
        return {
            'all_data_device': self.all_data.device(),
            'class_indices_device': self.class_indicies.device(),
            'all_data_size_mb': self.all_data.nbytes / 1024 / 1024
        }