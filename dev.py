import torch

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

data = stoi(text)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[:n]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size] for i in ix])
    return x, y
