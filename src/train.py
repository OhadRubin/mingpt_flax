
# !pip install datasets einops
# !wget https://huggingface.co/datasets/iohadrubin/TheRPTCollection/resolve/main/c4/c4-ice-in-pita-000-of-128.parquet
import os
os.environ["JAX_LOG_COMPILES"] = "1"
import sys
import json
import random
import jax
from ast import literal_eval
import jax.numpy as jnp
import numpy as np
import time
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader, IterableDataset
from tqdm.auto import tqdm
import optax
from flax.training.train_state import TrainState
from jax.nn import softmax
import einops
import math
from jax.numpy import where
import math

from flax import struct

class CfgNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)


from flax.linen.activation import tanh
import flax.linen as nn
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __call__(self, x):
        return 0.5 * x * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0))))




import flax
def init_Linear(out_features, bias=True, std=0.02):
    return flax.linen.Dense(features=out_features, use_bias=bias, dtype=jnp.float32, kernel_init=jax.nn.initializers.normal(std))



def init_Embedding(num_embeddings, embedding_dim, std=0.02):
    emb = flax.linen.Embed(num_embeddings, embedding_dim, embedding_init=jax.nn.initializers.normal(std))
    return emb





class CausalSelfAttention(nn.Module):
    config: CfgNode
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def setup(self):
        config=self.config
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = init_Linear(3 * config.n_embd) #init with std=0.02
        # output projection
        self.c_proj = init_Linear(config.n_embd, std=0.02/math.sqrt(2 * config.n_layer))
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.n_head = config.n_head
        self.n_embd = config.n_embd



    def __call__(self, x, deterministic: bool = True):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        bias = jnp.tril(jnp.ones([T, T]))[None,...][None,:]
        scaling_factor = jnp.ones(1,dtype=np.float32) / math.sqrt(self.n_embd//self.n_head)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v = self.rearrage_c_attn(self.c_attn(x))

        k,q,v = jax.tree_map(self.split_heads,[k,q,v])

        k = einops.rearrange(k, "... i j -> ... j i")
        att = (q @ k) * scaling_factor
        att = where(bias[:,:,:T,:T] == 0, float("-inf"), att)

        att = softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=deterministic)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = self.merge_heads(y)
        y = self.resid_dropout(self.c_proj(y), deterministic=deterministic)
        return y

    def split_heads(self,x):
        return einops.rearrange(x,"b t (n_head head_dim) -> b n_head t head_dim",n_head=self.n_head)

    def merge_heads(self,x):
        return einops.rearrange(x, "B nh T hs -> B T (nh hs)")

    @staticmethod
    def rearrage_c_attn(x):
        x = einops.rearrange(x,"b t (qkv n_embed) -> b t qkv n_embed",qkv=3)
        return einops.unpack(x,[[],[],[]],"b t * n_embed")

class MLP(nn.Module):
    config: CfgNode
    """ simple class for non-linear fully connected network """

    def setup(self):
        config=self.config
        self.c_fc = init_Linear(config.n_embd * 4)
        self.c_proj = init_Linear(config.n_embd, std=0.02/math.sqrt(2 * config.n_layer)) 
        self.act = NewGELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def __call__(self, x, deterministic: bool = True):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))), deterministic=deterministic)
    
class Block(nn.Module):
    """ an unassuming Transformer block """
    config: CfgNode

    def setup(self):
        config=self.config
        self.ln_1 = nn.LayerNorm()
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm()
        self.mlpf = MLP(config)

    def __call__(self, x, deterministic: bool = True):
        x = x + self.attn(self.ln_1(x), deterministic=deterministic)
        x = x + self.mlpf(self.ln_2(x), deterministic=deterministic)
        return x
class Timer:
    def __enter__(self):
        self.start = time.time()
        self.interval = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Elapsed time: {self.interval} seconds")



def setup_config(config):
    assert config.vocab_size is not None
    assert config.block_size is not None

    type_given = config.model_type is not None
    params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
    assert type_given ^ params_given # exactly one of these (XOR)
    if type_given:
        # translate from model_type to detailed configuration
        config.merge_from_dict({
            # names follow the huggingface naming conventions
            # GPT-1
            'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
            # GPT-2 configs
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
            # Gophers
            'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
            # (there are a number more...)
            # I made these tiny models up
            'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
            'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
            'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
        }[config.model_type])
    return config
        
class GPT(nn.Module):
    config: CfgNode
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CfgNode()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    

    def setup(self):
        config=self.config
        self.wte = init_Embedding(config.vocab_size, config.n_embd) #init with std=0.02
        self.wpe = init_Embedding(config.block_size, config.n_embd) #init with std=0.02
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        self.lm_head = init_Linear(config.vocab_size, bias=False) #init with std=0.02


    def __call__(self, idx, deterministic: bool = True):
        b, t = idx.shape
        pos = jnp.arange(0, t, dtype=jnp.int32)[None,...] # shape (1, t)
        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=deterministic)
        for block in self.h:
            x = block(x, deterministic=deterministic)
        x = self.ln_f(x)
        return self.lm_head(x)





class RollingAverage(struct.PyTreeNode):
  size: int
  last_element: int
  mat: np.ndarray
  discount: float = 0.99

  def update(self, new_value):
    mat=self.mat
    # mat = mat * self.discount
    mat[self.last_element] = new_value
    last_element = (self.last_element+1) % mat.shape[0]
    size = np.where(self.size!=mat.shape[0],self.size+1,self.size)
    curr_value = mat.sum()/ size
    # curr_value = mat.sum()/size

    return curr_value,self.replace(size=size,
                        last_element=last_element,
                        mat=mat,
                        )
  @classmethod
  def create(cls, *, size):
    return cls(size=0, last_element=0, mat=np.zeros(size,dtype=np.float32))



def train_step(state, input_idxs, targets, dropout_rng, axis='device'):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng, 2)
    def compute_loss(params):
        logits = state.apply_fn(dict(params=params), input_idxs, deterministic=False, rngs={"dropout":dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits,targets)
        # label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
        acc = (targets==logits.argmax(-1)).astype(jnp.float32)
        return jnp.mean(loss),jnp.mean(acc,axis=0)

    (loss, acc), grad = jax.value_and_grad(compute_loss,has_aux=True)(state.params)
    loss, grad, acc = jax.lax.pmean([loss, grad, acc], axis)

    new_state = state.apply_gradients(grads=grad)

    return loss, acc, new_state, new_dropout_rng

def eval_step(state, input_idxs, targets, dropout_rng, axis='device'):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng, 2)

    def compute_loss(params):
        logits = state.apply_fn(dict(params=params), input_idxs, deterministic=False, rngs={"dropout":dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits,targets)
        return jnp.mean(loss)
    loss = compute_loss(state.params)
    loss = jax.lax.pmean(loss, axis)
    return loss, state, new_dropout_rng
    
from flax import jax_utils
from flax.training.common_utils import shard



def calc_acc(logits, input_ids):
    shift_labels = input_ids[..., 1:]
    shift_logits = logits[..., :-1, :].argmax(-1)
    acc = (shift_labels==shift_logits).type(jnp.float32)
    return acc

class Trainer:

    @staticmethod
    def get_default_config():
        C = CfgNode()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.seed=42
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 8
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.warmup_steps = 1000
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)


        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
    def init_weights(self, rng: jax.random.PRNGKey, input_shape):
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")


        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.model.init(
            rngs,
            input_ids,
        )

        return module_init_outputs
        
    def run(self):
        model, config = self.model, self.config
        loss_metric = RollingAverage.create(size=20)
        p_train_step = jax.pmap(
            train_step,
            "device"
        )
        p_eval_step = jax.pmap(
            eval_step,
            "device"
        )
        
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
                    init_value=0,
                    peak_value=config.learning_rate,
                    warmup_steps=config.warmup_steps,
                    decay_steps=config.max_iters-config.warmup_steps,
                    end_value=config.learning_rate*0.1,
                )

        optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_norm_clip),
            optax.adamw(
            learning_rate=learning_rate_schedule,
            b1=config.betas[0],
            b2=config.betas[1],
            weight_decay=config.weight_decay,
        ))
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            # num_workers=1,
        )
        with Timer() as timer:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        
        
        rng = jax.random.PRNGKey(config.seed)
        dropout_rngs = jax.random.split(rng, jax.local_device_count())
        params = self.init_weights(rng, batch[0].shape[1:])
        print(params.keys())
        state = TrainState.create(apply_fn=model.apply, params=params["params"], tx=optimizer)
        
        state = jax_utils.replicate(state)

        self.iter_num = 0
        self.iter_time = time.time()
        pbar = tqdm()
        loss_metric = RollingAverage.create(size=20)
        acc0_metric = RollingAverage.create(size=20)
        acc1_metric = RollingAverage.create(size=20)
        acc2_metric = RollingAverage.create(size=20)
        acc3_metric = RollingAverage.create(size=20)

        while True:
            pbar.update(1)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            loss,acc, state, dropout_rngs = p_train_step(state, *batch, dropout_rngs)
            loss,acc = jax.device_get((loss,acc))
            
            acc_pos_num = acc.shape[-1]
            acc = acc.reshape([-1,acc_pos_num]).mean(0)
            num_buckets = 4
            acc = acc[-(num_buckets*(acc_pos_num//num_buckets)):]
            acc_per_pos = acc.reshape([num_buckets,acc_pos_num//num_buckets]).mean(1)
            acc0,acc1,acc2,acc3 = acc_per_pos

            curr_loss,loss_metric = loss_metric.update(loss.mean().item())
            curr_acc0,acc0_metric = acc0_metric.update(acc0.mean().item())
            curr_acc1,acc1_metric = acc1_metric.update(acc1.mean().item())
            curr_acc2,acc2_metric = acc2_metric.update(acc2.mean().item())
            curr_acc3,acc3_metric = acc3_metric.update(acc3.mean().item())
            #only show 2 decimal places
            # pbar.set_description(f"Curr loss: {curr_loss} Curr acc0: {curr_acc0} Curr acc1: {curr_acc1} Curr acc2: {curr_acc2} Curr acc3: {curr_acc3}")
            pbar.set_description(f"Curr loss: {curr_loss:.3f} Curr acc0: {curr_acc0:.3f} Curr acc1: {curr_acc1:.3f} Curr acc2: {curr_acc2:.3f} Curr acc3: {curr_acc3:.3f}")
            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
from datasets import load_dataset



import more_itertools
import numpy as np
def shift_right_by_one(arr,fill_val=2):
    shifted = np.roll(arr, shift=1, axis=0)
    shifted[0] = fill_val
    return shifted



import numpy as np
from copy import deepcopy
class BufferShuffledExamplesIterable:
    def __init__(self, ex_iterable, buffer_size: int, generator: np.random.Generator):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.buffer_size = buffer_size
        self.generator = generator
        # TODO(QL): implement iter_arrow

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size=1000):
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    def __iter__(self):
        buffer_size = self.buffer_size
        rng = deepcopy(self.generator)
        indices_iterator = self._iter_random_indices(rng, buffer_size)
        # this is the shuffle buffer that we keep in memory
        mem_buffer = []
        for x in self.ex_iterable:
            if len(mem_buffer) == buffer_size:  # if the buffer is full, pick and example from it
                i = next(indices_iterator)
                yield mem_buffer[i]
                mem_buffer[i] = x  # replace the picked example by a new one
            else:  # otherwise, keep filling the buffer
                mem_buffer.append(x)
        # when we run out of examples, we shuffle the remaining examples in the buffer and yield them
        rng.shuffle(mem_buffer)
        yield from mem_buffer
        
class IterableDatasetWrapper(IterableDataset):
    def __init__(self, dataset):
        super(IterableDatasetWrapper).__init__()
        self.dataset = dataset
    def __iter__(self):
        while True:
            for x in iter(self.dataset):
                yield x

import more_itertools
def flatten_input_ids(batch):
    out_list = []
    for y in more_itertools.chunked(batch["input_ids"],1024):
        y = np.array(y)
        out_list.append({"targets":y,"input_tokens":shift_right_by_one(y)})
    return out_list

def batched_parallel(dataset):
    dataset = iter(dataset)
    for x in dataset:
        yield from flatten_input_ids(x)



import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

def generate_sentences(sentence_lengths, word_distributions,topics,topic_idx_map):
    sentences = []
    vocab_size = word_distributions.shape[-1]
    sentence_topics = []
    for length, distribution,topic_idx in zip(sentence_lengths, word_distributions, topics):
        # Generate a sentence of 'length' words
        sentence = np.random.choice(vocab_size, length, p=distribution)
        sentences.extend(sentence)
        sentences.append(topic_idx_map[topic_idx])
        sentence_topics.extend([topic_idx] * length)


    return sentences, sentence_topics

def np_softmax(
    x,
    axis  = -1,
    where = None,
    initial = 0):
  x_max = np.max(x, axis, keepdims=True)
  unnormalized = np.exp(x - x_max)
  result = unnormalized / np.sum(unnormalized, axis, keepdims=True)
  return result

def sample_sequence(generator,n_sentences= 256, average_sentence_length = 5, dim_size=256, n_topics=64):
    vocab_size = 32
    topic_idx_map = {i:i+n_topics for i in range(n_topics)}
    topic_vectors = generator.normal(0,1,[n_topics,dim_size])
    word_emb = generator.normal(0,1,[vocab_size,dim_size])
    word_distributions_topic = np_softmax(topic_vectors@word_emb.T/10)
    topic_idx_sample = generator.integers(n_topics,size=[n_sentences])
    word_distributions_per_sentence = word_distributions_topic[topic_idx_sample]
    # sentence_lengths = np.random.poisson(average_sentence_length, n_sentences)
    sentence_lengths = [average_sentence_length for _ in range(n_sentences)]
    words, topics = generate_sentences(sentence_lengths,
                                       word_distributions_per_sentence,
                                       topic_idx_sample,
                                       topic_idx_map=topic_idx_map)
    return np.array(words), generator

from datasets import IterableDataset


def collate_fn(batch):
  targets = [x['targets'] for x in batch]
  input_tokens = [x['input_tokens'] for x in batch]
  targets = np.stack(targets)
  input_tokens = np.stack(input_tokens)
  return shard([input_tokens, targets])
def get_dataset():
    def gen(seed):
        generator = np.random.default_rng(seed)
        while True:
            input_ids,generator  = sample_sequence(generator)
            targets = shift_right_by_one(input_ids)
            yield {"input_tokens":input_ids,"targets":targets}
    dataset = IterableDataset.from_generator(gen,gen_kwargs={"seed":42})
    print(dataset)
    if isinstance(dataset,dict):
        dataset = dataset["train"]
    train_dataset = iter(dataset)
    # train_dataset = batched_parallel(train_dataset)
    # train_dataset = BufferShuffledExamplesIterable(tqdm(train_dataset,desc="prefetching"), buffer_size=10000, generator=np.random.default_rng(42))
    train_dataset = IterableDatasetWrapper(train_dataset)
    return train_dataset



import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

    
def go():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = 512 # openai's model vocabulary
    model_config.block_size = 4096  # openai's model block_size (i.e. input context length)
    model_config = setup_config(model_config)
    model = GPT(model_config)

    train_dataset = get_dataset()

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # many possible options, see the file
    train_config.max_iters = 100_000
    train_config.weight_decay = 0
    train_config.batch_size = 8
    trainer = Trainer(train_config, model, train_dataset)
    trainer.run()
    
if __name__ == "__main__":
    go()
