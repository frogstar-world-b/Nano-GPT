# a weighted average self-attention is obtained using
# - for loop
# - matrix multiplicaton
# - softmax


import torch
from torch.nn import functional as F
from torch import nn

torch.manual_seed(42)
# example of moving average using triangular ones matrix
a = torch.tril(torch.ones(3, 3))
# random numbers between 0 and 10 of shape (3, 2)
b = torch.randint(0, 10, (3, 2)).float()
# moving average becomes the dot product below
c = a @ b
print('a matrix:\n', a)
print('b matrix:\n', b)
print('Moving sum matrix c = a @ b:\n', c)

# normalize to get weights that sum to 1; keepdim=True to broadcast correctly
w = a / torch.sum(a, dim=1, keepdim=True)
d = w @ b
print('weights matrix w is normalized a:\n', w)
print('Moving average matrix d = w @ b:\n', d)
print('------------\n')

torch.manual_seed(1337)
# 4 sequences, each of length 8 tokens (time steps), and 2 channels
B, T, C = 4, 8, 2  # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

# A token should only be influenced by previous tokens in the sequence
# Example, we want the running average E(x[b, t]) = mean_{i<=t} x[b, i]

# bag of words initialized at zeros
xbow = torch.zeros((B, T, C))

for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]  # size (t, C)
        xbow[b, t] = torch.mean(xprev, dim=0)

print(x[0])
print(xbow[0])

weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(1, keepdim=True)
print('weights:\n', weights)
# (T, T) @ (B, T, C)
# PyTorch will create a batch dimention (B, T, T)
# And do in parallel (B, T, T) @ (B, T, C) ---> (B, T, C)
xbow2 = weights @ x
print(xbow2[0])
print('------------\n')
assert torch.allclose(xbow, xbow2)


tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
# tokens don't talk with the future
wei = wei.masked_fill(tril == 0, float('-inf'))
print('masked weights matrix:\n', wei)
# softmax calculate the probabilities which should sum to 1.
# this is done by exponentiating each value and dividing by a row's sum
wei = F.softmax(wei, dim=1)
print('softmax matrix:\n', wei)
xbow3 = wei @ x
print(xbow3[0])
assert torch.allclose(xbow, xbow3)

# single head of self-attention

head_size = 16
# nn.Linear applies a linear transformation y = x * transpose(beta) + bias
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B,T,16)
q = query(x)  # (B,T,16)
# need weights of size (T, T) at every batch
wei = q @ k.transpose(-2, -1)  # (B,T,16) @ (B,16,T) ---> (B,T,T)

tril = torch.tril(torch.ones(T, T))
# tokens don't talk with the future
wei = wei.masked_fill(tril == 0, float('-inf'))
# softmax calculate the probabilities which should sum to 1.
# this is done by exponentiating each value and dividing by a row's sum
wei = F.softmax(wei, dim=-1)
print(wei[0])

v = value(x)
out = wei @ v
print(out.shape)


def myfunc():
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    k = key(x)  # (B,T,16)
    q = query(x)  # (B,T,16)
    # print(key(x)[0,0,:])
    # print(query(x)[0,0,:])
    # print(value(x)[0,0,:])
    wei = q @ k.transpose(-2, -1)
    tril = torch.tril(torch.ones(T, T))
    # tokens don't talk with the future
    wei = wei.masked_fill(tril == 0, float('-inf'))
    # softmax calculate the probabilities which should sum to 1.
    # this is done by exponentiating each value and dividing by a row's sum
    wei = F.softmax(wei, dim=-1)
    print(wei[0])


m = nn.Linear(20, 30)
xb = torch.randn(128, 20)
output = m(xb)
print(output.size())
