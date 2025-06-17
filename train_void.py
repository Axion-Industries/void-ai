import torch
import pickle
import numpy as np
from model import GPTConfig, GPT

# Load the vocabulary and meta information
with open('data/void/vocab.pkl', 'rb') as f:
    chars, stoi = pickle.load(f)

with open('data/void/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

# Load and encode the input text
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# Train/val split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - meta['block_size'], (32,))
    x = torch.stack([data[i:i+meta['block_size']] for i in ix])
    y = torch.stack([data[i+1:i+1+meta['block_size']] for i in ix])
    return x, y

# Create the model
print("Initializing model...")
config = GPTConfig(
    vocab_size=meta['vocab_size'],
    block_size=meta['block_size'],
    n_layer=meta['n_layer'],
    n_head=meta['n_head'],
    n_embd=meta['n_embd'],
    dropout=meta['dropout']
)
model = GPT(config)

# Training settings
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
max_iters = 1000

print("Starting training...")
for iter in range(max_iters):
    if iter % 100 == 0:
        print(f"iteration {iter}/{max_iters}")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Saving model...")
torch.save(model.state_dict(), 'out/model.pt')
print("Done! Model saved to out/model.pt")
