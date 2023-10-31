import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32  # number of sequences in a batch
block_size = 8  # maximum context (max sequence length for prediction)
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embed = 32
n_heads = 4


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
        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadedAttention(nn.Module):
    '''Multiple heads of self-attention in parallel'''

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    '''a simple linear layer followed by non-linearity'''

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # get token embeddings; each token gets an embedding vector
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size,
                                                  embedding_dim=n_embed)
        # get position embeddings;
        # each position [0, block_size -1] gets an embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self-attention head
        self.ma_head = MultiHeadedAttention(n_heads, n_embed // n_heads)  # 4 heads of 8 dimentional self-attention
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
        x = self.ma_head(x)  # (B,T,C)
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

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()), '\n')


"""
Output below:

step 0: train loss 4.2338, val loss 4.2321
step 500: train loss 2.6220, val loss 2.6414
step 1000: train loss 2.4818, val loss 2.4916
step 1500: train loss 2.4267, val loss 2.4339
step 2000: train loss 2.3808, val loss 2.3963
step 2500: train loss 2.3615, val loss 2.3745
step 3000: train loss 2.3396, val loss 2.3605
step 3500: train loss 2.3185, val loss 2.3420
step 4000: train loss 2.3196, val loss 2.3106
step 4500: train loss 2.3112, val loss 2.3162
step 4999: train loss 2.3024, val loss 2.3170

UNTHEONG se tize herst els yurfuniexild.

Hak, CO ineg agrell torr Maecomy the dove?


ONGBUCAK:
Thoous of ingpery wallavend ou Thor wicires
Toknceanl-ht thy
Bou aw ely.

Thatiere, ale stselart'dcus;
I mer
Thavely ould mou farris, anrn itheland's of I gimy.

Q:
Tand gientyount they fouckile
Theureis, notrray nokesche
Tow men
Is, in mamy thouthelosmeroe, do of tthendy sak shil brveseardeaster him to, Tepomp rote dat thiming, thith Theandepsts lathise my:
Ber ourse I any son,
Kathicker.

FIINGLord
"""
