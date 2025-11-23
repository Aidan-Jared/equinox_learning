import functools
import sys
from collections.abc import Mapping

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from jaxtyping import Array, Float, Int
from tqdm import notebook as tqdm
from transformers import AutoTokenizer


class EmbedderBlock(eqx.Module):

    token_embedder: eqx.nn.Embedding
    segment_embedder: eqx.nn.Embedding
    position_embedder: eqx.nn.Embedding
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout


    def __init__(
            self,
            vocab_size: int,
            max_length: int,
            type_vocab_size: int,
            embedding_size: int,
            hidden_size: int,
            dropout_rate: float,
            key: jax.random.PRNGKey,
    ):
        token_key, segment_key, positon_key = jax.random.split(key, 3)
        
        self.token_embedder = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=embedding_size,
            key= token_key
        )

        self.segment_embedder = eqx.nn.Embedding(
            num_embeddings=type_vocab_size,
            embedding_size=embedding_size,
            key=segment_key
        )

        self.position_embedder = eqx.nn.Embedding(
            num_embeddings=max_length,
            embedding_size=embedding_size,
            key=positon_key
        )

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)

        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
            self,
            token_ids: Int[Array, " seq_len"],
            position_ids: Int[Array, " seq_len"],
            segment_ids: Int[Array, " seq_len"],
            enable_dropout: bool = False,
            key: jax.random.PRNGKey= None
    ) -> Float[Array, "seq_len hidden_size"]:
        tokens = jax.vmap(self.token_embedder)(token_ids)
        segments = jax.vmap(self.segment_embedder)(segment_ids)
        positions = jax.vmap(self.position_embedder)(position_ids)

        embedded_inputs = tokens + segments + positions

        embedded_inputs = jax.vmap(self.layernorm)(embedded_inputs)
        embedded_inputs = self.dropout(
            embedded_inputs, inference=not enable_dropout, key=key
        )
        return embedded_inputs
    
class FeedForwardBlock(eqx.Module):
    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            dropout_rate: float,
            key: jax.random.PRNGKey
    ):
        mlp_key, output_key = jax.random.split(key, 2)
        
        self.mlp = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            key=mlp_key
        )

        self.output = eqx.nn.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            key=output_key
        )

        self.layer_norm = eqx.nn.LayerNorm(shape=hidden_size)

        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
            self,
            inputs: Float[Array, " hidden_size"],
            enable_dropout: bool = True,
            key: jax.random.PRNGKey = None
    ) -> Float[Array, " hidden_size"]:
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)
        
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        output += inputs
        output = self.layer_norm(output)

        return output
        
    
class AttentionBlock(eqx.Module):

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            dropout_rate: float,
            attention_dropout_rate: float,
            key = jax.random.PRNGKey,
    ):
        self.num_heads = num_heads

        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key
        )

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
            self,
            inputs: Float[Array, "seq_len hidden_size"],
            mask: Int[Array, " seq_len"],
            enable_dropout: bool = False,
            key: "jax.random.PRNGKey" = None
    ) -> Float[Array, "seq_len hidden_size"]:
        if mask is not None:
            mask = self.make_self_attention_mask(mask)
        
        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key,2)
        )

        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            inference= not enable_dropout,
            key = attention_key
        )

        result = attention_output

        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)

        result = result + inputs

        result = jax.vmap(self.layernorm)(result)

        return result
    
    def make_self_attention_mask(
            self,
            mask: Int[Array, " seq_len"]
    ) -> Float[Array, "num_heads seq_len seq_len"]:
        mask = jnp.multiply(
            jnp.expand_dims(mask, axis=-1),
            jnp.expand_dims(mask, axis=-2)
        )

        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)

class TransformerLayer(eqx.Module):

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            num_heads: int,
            dropout_rate: float,
            attention_dropout_rate: float,
            key: jax.random.PRNGKey
    ):
        
        attention_key, ff_key = jax.random.split(key, 2)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key
        )

        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key
        )
    
    def __call__(
            self,
            inputs: Float[Array, "seq_len hidden_size"],
            mask: Int[Array, " seq_len"] = None,
            *,
            enable_dropout: bool = False,
            key: jax.random.PRNGKey = None
    ) -> Float[Array, " seq_len hidden_size"]:
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key, 2)

        attention_output = self.attention_block(
            inputs,
            mask,
            enable_dropout=enable_dropout,
            key= attn_key
        )

        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)

        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )

        return output

class Encoder(eqx.Module):

    embedder_block: EmbedderBlock
    layers: list[TransformerLayer]
    pooler: eqx.nn.Linear

    def __init__(
            self,
            vocab_size: int,
            max_length: int,
            type_vocab_size: int,
            embedding_size: int,
            hidden_size: int,
            intermediate_size: int,
            num_layers: int,
            num_heads: int,
            dropout_rate: float,
            attentioin_dropout_rate: float,
            key: jax.random.PRNGKey
    ):
        embedder_key, layer_key, pooler_key = jax.random.split(key, 3)

        self.embedder_block = EmbedderBlock(
            vocab_size=vocab_size,
            max_length=max_length,
            type_vocab_size=type_vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            key=embedder_key
        )

        layer_keys = jax.random.split(layer_key, num_layers)

        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attentioin_dropout_rate,
                    key=layer_key
                )
            )

        self.pooler = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            key=pooler_key
        )

    def __call__(
            self,
            token_ids: Int[Array, " seq_len"],
            position_ids: Int[Array, " seq_len"],
            segment_ids: Int[Array, " seq_len"],
            *,
            enable_dropout: bool = False,
            key: jax.random.PRNGKey = None
    ) -> dict[str, Array]:
        emb_key, l_key = (None, None) if key is None else jax.random.split(key, 2)

        embeddings = self.embedder_block(
            token_ids=token_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
            enable_dropout=enable_dropout,
            key=emb_key
        )

        mask = jnp.asarray(token_ids != 0, dtype=jnp.int32)

        x = embeddings

        layer_outputs = []

        for layer in self.layers:
            cl_key, l_key = (None, None) if l_key is None else jax.random.split(l_key, 2)
            x = layer(x, mask, enable_dropout=enable_dropout, key=cl_key)
            layer_outputs.append(x)
        
        first_token_last_layer = x[..., 0, :]
        pooled = self.pooler(first_token_last_layer)
        pooled = jnp.tanh(pooled)

        return {"embeddings": embeddings, "layers": layer_outputs, "pooled": pooled}
    

class BertClassifier(eqx.Module):
    encoder:Encoder
    classifier_head: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
            self,
            config: Mapping,
            num_classes: int,
            key: jax.random.PRNGKey
    ):
        
        encoder_key, head_key = jax.random.split(key, 2)

        self.encoder = Encoder(
            vocab_size=config["vocab_size"],
            max_length=config["max_position_embeddings"],
            type_vocab_size=config["type_vocab_size"],
            embedding_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            dropout_rate=config["hidden_dropout_prob"],
            attentioin_dropout_rate=config["attention_probs_dropout_prob"],
            key= encoder_key
        )

        self.classifier_head = eqx.nn.Linear(
            in_features= config["hidden_size"], out_features=num_classes, key=head_key
        )
        self.dropout = eqx.nn.Dropout(config["hidden_dropout_prob"])

    def __call__(
            self,
            inputs: dict[str, Int[Array, " seq_len"]],
            enable_dropout: bool = True,
            key: jax.random.PRNGKey = None
    ) -> Float[Array, " num_classes"]:
        seq_len = inputs["token_ids"].shape[-1]
        position_ids = jnp.arange(seq_len)
        
        e_key, d_key = (None, None) if key is None else jax.random.split(key, 2)

        pooled_output = self.encoder(
            token_ids=inputs["token_ids"],
            segment_ids=inputs["segment_ids"],
            position_ids=position_ids,
            enable_dropout=enable_dropout,
            key=e_key
        )["pooled"]

        pooled_output = self.dropout(
            pooled_output, inference=not enable_dropout, key=d_key
        )

        return self.classifier_head(pooled_output)

def tokenize(example):
    return tokenizer(example["sentence"], padding="max_length",  truncation=True)

@eqx.filter_value_and_grad
def compute_loss(classifier: BertClassifier, inputs, key):
    batch_size = inputs["token_ids"].shape[0]
    batched_keys = jax.random.split(key, batch_size)
    logits = jax.vmap(classifier, in_axes=(0, None, 0))(inputs, True,batched_keys)

    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=inputs["label"]
        )
    )

@eqx.filter_jit
def make_step(model, inputs, opt_state, key, tx):
    key, new_key = jax.random.split(key, 2)
    loss, grads = compute_loss(model, inputs, key)
    grads = jax.lax.pmean(grads, axis_name="devices")

    updates, opt_state = tx.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state, new_key

@eqx.filter_pmap
def make_eval_step(model, inputs):
    return jax.vmap(functools.partial(model, enable_dropout = False))(inputs)


if __name__ == "__main__":
    bert_config = {
        "vocab_size": 30522,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_act": "gelu",
        "intermediate_size": 512,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
    }

    key = jax.random.PRNGKey(42)

    model_key, train_key = jax.random.split(key)

    classifier = BertClassifier(config=bert_config, num_classes=2, key=model_key)

    # classifier_chkpt = eqx.tree_deserialise_leaves("bert_checkpoint.eqx", classifier)

    tokenizer = AutoTokenizer.from_pretrained(
        "google/bert_uncased_L-2_H-128_A-2", model_max_length=128
    )

    ds = load_dataset("sst2")

    ds = ds.map(tokenize, batched=True)

    ds.set_format(type="jax", columns=["input_ids", "token_type_ids", "label"])

    epochs = 3
    batch_size = 32
    learning_rate = 1e-5

    num_devices = jax.device_count()

    assert batch_size % num_devices == 0, (
        "The batch size must be a multiple of the number of devices."
    )

    tx = optax.adam(learning_rate=learning_rate)
    tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
    opt_state = tx.init(classifier)

    p_make_step = eqx.filter_pmap(functools.partial(make_step, tx=tx), axis_name="devices")

    opt_state = jax.device_put_replicated(opt_state, jax.local_devices())
    model = jax.device_put_replicated(classifier, jax.local_devices())
    train_key = jax.device_put_replicated(train_key, jax.local_devices())

    for epoch in range(epochs):
        train_iter = ds["train"].iter(batch_size=batch_size, drop_last_batch=True)
        with tqdm.tqdm(
            train_iter,
            total=ds["train"].num_rows // batch_size,
            unit="steps",
            desc=f"Epoch {epoch + 1}/{epochs}",
            mininterval=0,
            miniters=1.
        ) as tqdm_epoch:
            for batch in tqdm_epoch:
                token_ids, token_type_ids = batch["input_ids"], batch["token_type_ids"]
                label = batch['label']

                token_ids = einops.rearrange(
                    token_ids, "(b1 b2) s -> b1 b2 s", b1 = num_devices
                )

                token_type_ids = einops.rearrange(
                    token_type_ids, "(b1 b2) s -> b1 b2 s", b1 = num_devices
                )

                label = einops.rearrange(
                    label, "(b1 b2) -> b1 b2", b1 = num_devices
                )

                inputs = {
                    "token_ids": token_ids,
                    "segment_ids": token_type_ids,
                    "label": label
                }

                loss, model, opt_state, train_key = p_make_step(
                    model, inputs, opt_state, train_key
                )

                tqdm_epoch.set_postfix(loss = np.sum(loss).item(), refresh=True)
    
    outputs = []

    for batch in tqdm.tqdm(
        ds["validation"].iter(batch_size=batch_size),
        unit="steps",
        total = np.ceil(ds["validation"].num_rows / batch_size),
        desc= "Validation",
    ):
        token_ids, token_type_ids = batch["input_ids"], batch["token_type_ids"]
        label = batch["label"]

        token_ids = einops.rearrange(
                    token_ids, "(b1 b2) s -> b1 b2 s", b1 = num_devices
                )

        token_type_ids = einops.rearrange(
            token_type_ids, "(b1 b2) s -> b1 b2 s", b1 = num_devices
        )

        label = einops.rearrange(
            label, "(b1 b2) -> b1 b2", b1 = num_devices
        )

        inputs = {
            "token_ids": token_ids,
            "segment_ids": token_type_ids,
            "label": label
        }

        output = make_eval_step(model, inputs)
        output = (np.argmax(output.reshape(-1, 2), axis=-1) == label).tolist()[0]

        outputs.extend(output)
    print(f"Accuracy: {100 * np.sum(outputs) / len(outputs):.2f}%")