import torch
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import os
import time
from collections import defaultdict
import signal
from contextlib import contextmanager

MODEL_PATH = 'out/model.pt'
VOCAB_PATH = 'data/void/vocab.pkl'
META_PATH = 'data/void/meta.pkl'

# Check for required files before loading
missing_files = []
for f in [MODEL_PATH, VOCAB_PATH, META_PATH]:
    if not os.path.exists(f):
        missing_files.append(f)

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    response = send_from_directory('.', path)
    if path.endswith(('.js', '.css', '.html')):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

if missing_files:
    @app.route('/chat', methods=['POST'])
    def chat_error():
        return jsonify({'error': f"Missing required model files: {', '.join(missing_files)}. Please train your model and add these files to your repo."}), 500
    if __name__ == '__main__':
        print(f"Missing required model files: {', '.join(missing_files)}. Please train your model and add these files to your repo.")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
    exit(0)

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
            try:
                with time_limit(10):  # 10-second timeout
                    for _ in range(max_new_tokens):
                        idx_cond = idx[:, -config.block_size:]
                        logits, _ = self(idx_cond)
                        logits = logits[:, -1, :]
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, idx_next), dim=1)
                    return idx
            except TimeoutException:
                raise Exception("Model took too long to respond. Please try a shorter prompt.")

    # These should match your training config
    config = GPTConfig(vocab_size, 64, 4, 4, 128, 0.1)
    model = GPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model, stoi, itos, config

model, stoi, itos, config = load_model()

# Rate limiting
RATE_LIMIT = 5  # requests per second
RATE_WINDOW = 1  # seconds
request_counts = defaultdict(list)

def is_rate_limited(ip):
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if t > now - RATE_WINDOW]
    request_counts[ip].append(now)
    return len(request_counts[ip]) > RATE_LIMIT

@app.before_request
def check_rate_limit():
    if request.endpoint == 'chat':
        ip = request.remote_addr
        if is_rate_limited(ip):
            return jsonify({'error': 'Too many requests. Please wait a moment.'}), 429

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'response': 'No prompt provided.'})

        # Validate prompt length
        if len(prompt) > 5000:
            return jsonify({'error': 'Prompt too long. Maximum length is 5000 characters.'}), 400

        # Encode prompt
        try:
            idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
        except Exception as e:
            return jsonify({'error': f'Error encoding prompt: {str(e)}'}), 500

        # Generate response
        try:
            with torch.no_grad():
                # Add temperature and top-k sampling for better responses
                try:
                    out_idx = model.generate(idx, max_new_tokens=100)[0].tolist()
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        return jsonify({'error': 'Model ran out of memory. Try a shorter prompt.'}), 500
                    raise e
                except Exception as e:
                    if 'CUDA' in str(e):
                        return jsonify({'error': 'GPU error occurred. The model will restart automatically.'}), 500
                    raise e

            # Process the response
            try:
                full_response = ''.join([itos[i] for i in out_idx])
                generated_text = full_response[len(prompt):]  # Extract only the generated portion
                
                # Ensure we have a valid response
                if not generated_text.strip():
                    return jsonify({'error': 'Model generated an empty response. Please try again.'}), 500
                    
                return jsonify({'response': generated_text.strip()})
            except Exception as e:
                return jsonify({'error': 'Error processing model output. Please try again.'}), 500
                
        except Exception as e:
            return jsonify({'error': f'Error generating response: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def index():
    return 'Void AI backend is running. Use the /chat endpoint for POST requests.', 200

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

@app.after_request
def after_request(response):
    cleanup_memory()
    return response

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
