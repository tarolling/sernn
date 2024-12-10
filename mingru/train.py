import math
import gzip
import random
import tqdm
import numpy as np
from typing import Any

import tensorflow as tf
import jax.numpy as jnp
import keras
from keras import layers

from minGRULM import minGRULM

tf.random.set_seed(42)

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 2
GRAD_ACCUM_EVERY = 2
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 1
PRIME_LENGTH = 128
GENERATE_EVERY = 1
GENERATE_LENGTH = 512
SEQ_LEN = 512


# helpers
def exists(v: Any):
    return v is not None


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# sampling helpers


def log(t: tf.Tensor, eps=1e-20) -> tf.Tensor:
    return jnp.log(jnp.clip(t, min=eps))


def gumbel_noise(t: tf.Tensor) -> tf.Tensor:
    noise = tf.random.uniform(t.shape, minval=0, maxval=1)
    return -log(-log(noise))


def gumbel_sample(t: tf.Tensor, temperature=1.0, dim=-1, keepdim=True) -> tf.Tensor:
    sample = (t / max(temperature, 1e-10)) + gumbel_noise(t)
    return tf.math.argmax(sample, axis=dim)


def top_k(logits: tf.Tensor, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = tf.math.top_k(logits, k)
    return tf.scatter_nd(ind, val, logits.shape)


def base_decoding(
    net,
    prompt: tf.Tensor,
    seq_len: int,
    temperature=1.0,
    filter_thres=0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    prev_hiddens = None

    for _ in range(sample_num_times):
        logits, next_prev_hiddens = net(
            out, return_prev_hiddens=True, prev_hiddens=prev_hiddens
        )
        logits = logits[:, -1]

        if net.can_cache:
            prev_hiddens = next_prev_hiddens

        logits = top_k(logits, thres=filter_thres)
        sample = gumbel_sample(logits, temperature=temperature, dim=-1)
        out = tf.concat([out, sample], axis=-1)

    return out[..., prompt_seq_len:]


# the minGRU char language model

model = minGRULM(num_tokens=256, input_dim=512, depth=6)
model.summary()

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = tf.convert_to_tensor(np_train), tf.convert_to_tensor(
        np_valid
    )


# class TextSamplerDataset(tf.data.Dataset):
#     def __init__(self, data, seq_len):
#         super().__init__()
#         self.data = data
#         self.seq_len = seq_len

#     def __len__(self):
#         return self.data.size(0) // self.seq_len

#     def __getitem__(self, index):
#         rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
#         full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
#         return full_seq.cuda()


# train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
# val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# # optimizer

# optim = Adam(model.parameters(), lr=LEARNING_RATE)

# train_loader = cycle(train_loader)
# val_loader = cycle(val_loader)

# # training

# for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
#     model.train()

#     for _ in range(GRAD_ACCUM_EVERY):
#         data = next(train_loader)

#         loss = model(data, return_loss=True)

#         (loss / GRAD_ACCUM_EVERY).backward()

#     print(f"training loss: {loss.item():.3f}")

#     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

#     optim.step()
#     optim.zero_grad()

#     if i % VALIDATE_EVERY == 0:
#         model.eval()
#         with torch.no_grad():
#             valid_data = next(val_loader)

#             loss = model(valid_data, return_loss=True)
#             print(f"validation loss: {loss.item():.3f}")

#     if i % GENERATE_EVERY == 0:
#         model.eval()

#         inp = random.choice(val_dataset)[:PRIME_LENGTH]
#         inp = inp.cuda()

#         prime = decode_tokens(inp)
#         print(f"INPUT: {prime}")

#         prompt = inp[None, ...]

#         sampled = base_decoding(model, prompt, GENERATE_LENGTH)

#         base_decode_output = decode_tokens(sampled[0])

#         print(f"\nOUTPUT: {base_decode_output}")
