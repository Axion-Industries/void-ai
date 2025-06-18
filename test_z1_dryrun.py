"""
Void Z1 dry-run test script: checks model loading, prompt inference, and output.
"""
import torch
import pickle
from model import GPT, GPTConfig

VOCAB_PATH = 'data/void/vocab.pkl'
META_PATH = 'data/void/meta.pkl'
MODEL_PATH = 'out/model.pt'

with open(VOCAB_PATH, 'rb') as f:
    chars, stoi = pickle.load(f)
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)
with open(META_PATH, 'rb') as f:
    meta = pickle.load(f)
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=meta.get('block_size', 64),
    n_layer=meta.get('n_layer', 4),
    n_head=meta.get('n_head', 4),
    n_embd=meta.get('n_embd', 128),
    dropout=meta.get('dropout', 0.1),
    bias=meta.get('bias', True)
)
model = GPT(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

prompt = "Hello, Z1!"
idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
with torch.no_grad():
    for _ in range(50):
        logits, _ = model(idx)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
output = ''.join([itos[i] for i in idx[0].tolist()])
print('Prompt:', prompt)
print('Z1 Output:', output[len(prompt):].strip())
