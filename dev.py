from bigramLanguageModel import BigramLanguageModel
import torch
from tqdm import tqdm

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('length of text: ', len(text))

# print(text[:1000])
chars = sorted(list(set(text)))
vocabulary_size = len(chars)
print(''.join(chars))
print('vocabulary size: ', vocabulary_size)

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}


def encode(input): return [stoi[i] for i in input]
def decode(input): return ''.join([itos[i] for i in input])


print(encode('hello world'))
print(decode(encode('hello world')))

torch.manual_seed(1337)
batch_size = 4
block_size = 8

data = encode(text)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[:n]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.tensor([data[i:i+block_size] for i in ix])
    y = torch.tensor([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")


m = BigramLanguageModel(vocabulary_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx,
      max_new_token=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in tqdm(range(10000)):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx,
      max_new_token=500)[0].tolist()))
