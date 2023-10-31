import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32  # number of sequences in a batch
block_size = 8  # maximum context (max sequence length for prediction)
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)


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
encode = lambda s: [stoi[c] for c in s]
# decode a list of integers into a string
decode = lambda l: ''.join([itos[i] for i in l])


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


# simple bigram model
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # nn.Module.Embedding is a module for word embeddings.
        # Word embeddings are dense vector representations of words,
        # which are often used as a foundational component in NLP tasks
        # see https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # You need to specify at least num_embeddings and embedding_dim
        # that is, when you create an instance of nn.Embedding,
        # you specify the number of unique words in your vocabulary
        # (i.e., the vocabulary size) and the dimension of the dense
        # embedding vectors
        # If you plan to use pre-trained word embeddings (e.g., Word2Vec, GloVe
        # or FastText), the embedding dimension should match the dimensionality
        # of the pre-trained embeddings you are using
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size,
                                                  embedding_dim=vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers

        # logits size is: (B, T, C)
        # B: batch_size
        # T: Time aka sequence length or block_size
        # C: number of classes
        logits = self.token_embedding_table(idx)
        B, T, C = logits.shape
        if targets is None:
            loss = None
        else:
            # torch.nn.functional.cross_entropy requires size (N, C)
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
            # idx is (B, T) arry of indices in the current context
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilies
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLM(vocab_size)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
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
output below:

step 0: train loss 4.7305, val loss 4.7241
step 300: train loss 2.8110, val loss 2.8249
step 600: train loss 2.5434, val loss 2.5682
step 900: train loss 2.4932, val loss 2.5088
step 1200: train loss 2.4863, val loss 2.5035
step 1500: train loss 2.4665, val loss 2.4921
step 1800: train loss 2.4683, val loss 2.4936
step 2100: train loss 2.4696, val loss 2.4846
step 2400: train loss 2.4638, val loss 2.4879
step 2700: train loss 2.4738, val loss 2.4911

Foasthaprse tize herst el
O u fZEie hy:


Hak, CORineg aggell thrr Masearor charnge?
Tyoucre thy, chouspo in mppry way avend oubur'er sickes bokecard dhiceny

He tw el fe oupise he, lbustselownthous;
I m w
T:
The at;
I m hofaruk mondrn itheland's oe, oghithet f, badogienthofBRI'sey &CleDWeer'dsureisold array n
ICoyockind m murs, in mamybalorenyongmyooe, d Vofetthindy st
Hefqu brveseay alsteanerm to, oupomp rede d pre h, gavitYOfrrerean apsts lathind my d erouerse IOLUED d ngKE hicerire.
II IS:
I 
"""