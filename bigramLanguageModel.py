import torch
import torch.nn as nn
from torch.nn import functional as F
import attention
torch.manual_seed(1337)


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = attention.MultiHeadAttention(
            n_head, head_size, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size,
                 device, n_layer, dropout, n_head):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout)
              for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.device = device
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targtets are both (B, T)
        token_embed = self.token_embedding_table(idx)  # (B, T, C)
        position_embed = self.position_embedding_table(
            torch.arange(T, device=self.device))  # (T,C)
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            idx_crop = idx[:, -self.block_size:]
            logits, loss = self(idx_crop)
            logits = logits[:, -1, :]  # take last one, the pred
            probs = F.softmax(logits, dim=-1)
            # choose one from prob (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
