import torch
import pickle
import numpy as np
from flask import Flask, request, jsonify
import os

MODEL_PATH = 'out/model.pt'
VOCAB_PATH = 'data/void/vocab.pkl'
META_PATH = 'data/void/meta.pkl'

# Check for required files before loading
missing_files = []
for f in [MODEL_PATH, VOCAB_PATH, META_PATH]:
    if not os.path.exists(f):
        missing_files.append(f)
if missing_files:
    raise FileNotFoundError(f"Missing required model files: {', '.join(missing_files)}. Please train your model and add these files to your repo.")

# Load model and vocab
def load_model():
    with open(VOCAB_PATH, 'rb') as f:
        chars, stoi = pickle.load(f)
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(chars)

    class GPTConfig:
        def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
            self.vocab_size = vocab_size
            self.block_size = block_size
            self.n_layer = n_layer
            self.n_head = n_head
            self.n_embd = n_embd
            self.dropout = dropout

    class Head(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.key = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.query = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.value = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)
            self.dropout = torch.nn.Dropout(config.dropout)
            self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

        def forward(self, x):
            B,T,C = x.size()
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2,-1) * C**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = torch.nn.functional.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            v = self.value(x)
            out = wei @ v
            return out

    class MultiHeadAttention(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.heads = torch.nn.ModuleList([Head(config) for _ in range(config.n_head)])
            self.proj = torch.nn.Linear(config.n_embd, config.n_embd)
            self.dropout = torch.nn.Dropout(config.dropout)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out

    class FeedForward(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(config.n_embd, 4*config.n_embd),
                torch.nn.ReLU(),
                torch.nn.Linear(4*config.n_embd, config.n_embd),
                torch.nn.Dropout(config.dropout),
            )
        def forward(self, x):
            return self.net(x)

    class Block(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.sa = MultiHeadAttention(config)
            self.ffwd = FeedForward(config)
            self.ln1 = torch.nn.LayerNorm(config.n_embd)
            self.ln2 = torch.nn.LayerNorm(config.n_embd)

        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class GPT(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.token_embedding_table = torch.nn.Embedding(config.vocab_size, config.n_embd)
            self.position_embedding_table = torch.nn.Embedding(config.block_size, config.n_embd)
            self.blocks = torch.nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
            self.ln_f = torch.nn.LayerNorm(config.n_embd)
            self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size)

        def forward(self, idx, targets=None):
            B,T = idx.shape
            token_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
            x = token_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

            if targets is None:
                loss = None
            else:
                B,T,C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = torch.nn.functional.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

    # These should match your training config
    config = GPTConfig(vocab_size, 64, 4, 4, 128, 0.1)
    model = GPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model, stoi, itos, config

model, stoi, itos, config = load_model()
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'response': 'No prompt provided.'})
    # Encode prompt
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
    with torch.no_grad():
        out_idx = model.generate(idx, max_new_tokens=100)[0].tolist()
    response = ''.join([itos[i] for i in out_idx])
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
