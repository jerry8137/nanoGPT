import torch
from bigramLanguageModel import BigramLanguageModel
from tqdm import tqdm

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('length of text: ', len(text))

# print(text[:1000])
chars = sorted(list(set(text)))
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}


def encode(input): return [stoi[i] for i in input]
def decode(input): return ''.join([itos[i] for i in input])


data = encode(text)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[:n]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.tensor([data[i:i+block_size] for i in ix], device=device)
    y = torch.tensor([data[i+1:i+block_size+1] for i in ix], device=device)
    return x, y


xb, yb = get_batch('train')
model = BigramLanguageModel(len(chars), n_embed)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        output[split] = loss.mean()
    model.train()
    return output


for steps in (range(max_iters)):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % eval_interval == 0:
        eval_loss = estimate_loss()
        print(f"step: {steps}:")
        print(f"  train loss: {eval_loss['train']:.4f},")
        print(f"  val loss: {eval_loss['val']:.4f}")

print(loss.item())

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_token=500)[0].tolist()))
