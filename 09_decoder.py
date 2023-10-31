import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64  # number of sequences in a batch
block_size = 256  # maximum context (max sequence length for prediction)
max_iters = 5000
eval_interval = 500
learning_rate = 5e-4
eval_iters = 200
n_embed = 384
n_heads = 6
n_layers = 6  # number of transformer layers
dropout = 0.2

device = torch.device('mps')
print(device)

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters
chars = sorted(list(set(text)))
# vocabulary size
vocab_size = len(chars)
# mappings from char to integers and visa versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encode a string to a list of integers
def encode(s): return [stoi[c] for c in s]
# decode a list of integers into a string
def decode(l): return ''.join([itos[i] for i in l])


# train and validation splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% of the data
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    # draw the starting indices of the sequences in a batch
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# The @torch.no_grad() decorator is used as a context manager in here
# to temporarily disable gradient computation (and back propagation) during
# the execution of the estimate_loss function.
@torch.no_grad()
def estimate_loss(model):
    '''Averages out the loss over multiple batches.
    '''
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    '''Single head of self-attention'''

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # we want a lower triangular matrix variable but since it not
        # a model parameters, pytorch requires assignmet w/ registered_buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size,
                                                           block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.key(x)
        # compute attention scores / affinities: (B,T,C)@(B,C,T)--->(B,T,T)
        weights = q @ k.transpose(-2, -1) * C**-0.5
        # make it a decoder block (a token only talks with the past)
        weights = weights.masked_fill(self.tril[:T, :T] == 0,
                                      float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadedAttention(nn.Module):
    '''Multiple heads of self-attention in parallel'''

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # the projection is a linear transform of the outcome of the sa layer
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    '''a simple linear layer followed by non-linearity. Works on
    per-token level. The attetion does the communication. This
    does the computations.'''

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # paper uses 4x for inner layer
            nn.ReLU(),
            # projection layer going back into residual pathway
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    '''Transformer block: communication followed by computation.
    Aka message passing between tokens then computation.'''

    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadedAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        '''include residual connections.'''
        x = x + self.sa(self.ln1(x)
                        )  # fork off, do some computations, then come back
        # fork off, do some computations, then come back
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # get token embeddings; each token gets an embedding vector
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size,
                                                  embedding_dim=n_embed)
        # get position embeddings;
        # each position [0, block_size -1] gets an embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # transformer block
        self.tblocks = nn.Sequential(
            *[TransformerBlock(n_embed, n_heads) for _ in range(n_layers)]
        )
        # final norm layer
        self.ln_f = nn.LayerNorm(n_embed)
        # need a linear layer to get logits; lm_head for language model head
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C=n_embed)
        # integers 0 to T-1 get embedded through the position_embedding_table
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C=n_embed)
        x = token_embeddings + position_embeddings  # (B,T,C) from broadcasting
        x = self.tblocks(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B, T, C=vocab_size)
        B, T, C = logits.shape
        if targets is None:
            loss = None
        else:
            # torch.nn.functional.cross_entropy requires size (batch_size,C)
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        '''Take the idx sequence which is (B, T) and extend it
        sequentially in the time dimention to (B, T+1), (B, T+2), ...
        and up to max_new_tokens.
        '''
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens b/c our positional
            # encoding only works for up to block_size
            idx_crop = idx[:, -block_size:]
            # idx is (B, T) arry of indices in the current context
            logits, loss = self(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilies
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPTLanguageModel()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(model)
# number of model parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters: {num_params}")


for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model using a single  kickoff token ("\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
open('out2k.txt', 'w').write(
    decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
print('All Done!!!')

# generate from the model using a kickoff sequence
# kickoff_text = "First Citizen:\nLet us be brave, and we'll be rewarded with victory.\n"
# kickoff_context = torch.tensor(encode(kickoff_text),
#                                device=device).view(1, len(kickoff_text))
# print(decode(model.generate(kickoff_context, max_new_tokens=500)[0].tolist()), '\n')


"""
Output includes:

Number of model parameters: 10788929

step 0: train loss 4.4736, val loss 4.4690
step 500: train loss 2.1586, val loss 2.2122
step 1000: train loss 1.6820, val loss 1.8467
step 1500: train loss 1.4781, val loss 1.6718
step 2000: train loss 1.3685, val loss 1.5945
step 2500: train loss 1.2977, val loss 1.5506
step 3000: train loss 1.2447, val loss 1.5173
step 3500: train loss 1.2018, val loss 1.4992
step 4000: train loss 1.1629, val loss 1.4954
step 4500: train loss 1.1291, val loss 1.4834
step 4999: train loss 1.0954, val loss 1.4967
"""
