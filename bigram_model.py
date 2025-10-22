import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

# Read the input text
with open("input.txt", 'r', encoding="utf-8") as f:
    text = f.read()

# Create vocabulary from unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create encoding and decoding mappings
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [string_to_int[c] for c in s]

def decode(l):
    return ''.join([int_to_string[i] for i in l])

# Convert text to tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split into train and validation sets
n = int(0.9 * len(text))
train_data = data[:n]
val_data = data[n:]

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # note: it is (B,T,C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # len(data) - block_size is the maximum index we can start from
    random_indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in random_indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_indices])
    return x, y

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    block_size = 8
    torch.manual_seed(1337)

    # Initialize model
    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    for steps in range(10000):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate some text
    print("Generated text:")
    print(decode(model.generate(idx=torch.zeros((1, 1), dtype=torch.long), 
                              max_new_tokens=300)[0].tolist()))