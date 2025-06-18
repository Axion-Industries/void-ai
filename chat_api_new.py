import torch
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import os

# Use absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'out', 'model.pt')
VOCAB_PATH = os.path.join(BASE_DIR, 'data', 'void', 'vocab.pkl')
META_PATH = os.path.join(BASE_DIR, 'data', 'void', 'meta.pkl')

app = Flask(__name__, static_folder='.', static_url_path='')

def load_model():
    # Load vocabulary
    with open(VOCAB_PATH, 'rb') as f:
        chars, stoi = pickle.load(f)
    vocab_size = len(chars)
    itos = {i: ch for ch, i in stoi.items()}
    
    # Define model architecture
    class GPT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, 128)
            self.lstm = torch.nn.LSTM(128, 128, num_layers=4, batch_first=True)
            self.fc = torch.nn.Linear(128, vocab_size)
            
        def forward(self, idx, targets=None):
            x = self.embedding(idx)
            x, _ = self.lstm(x)
            logits = self.fc(x)
            
            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = torch.nn.functional.cross_entropy(logits, targets)
            
            return logits, loss
        
        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                # Get predictions
                logits, _ = self(idx)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
    
    # Create and load the model
    model = GPT()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    return model, stoi, itos

# Check if all required files exist
missing_files = []
for f in [MODEL_PATH, VOCAB_PATH, META_PATH]:
    if not os.path.exists(f):
        missing_files.append(f)

if missing_files:
    @app.route('/chat', methods=['POST'])
    def chat_error():
        return jsonify({
            'error': f"Missing required model files: {', '.join(missing_files)}. Please train your model and add these files to your repo."
        }), 500
else:
    # Load the model if all files exist
    model, stoi, itos = load_model()
    
    @app.route('/chat', methods=['POST'])
    def chat():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            prompt = data.get('prompt', '')
            if not prompt:
                return jsonify({'response': 'No prompt provided.'})
            
            # Encode the input text
            idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
            
            # Generate response
            with torch.no_grad():
                out_idx = model.generate(idx, max_new_tokens=100)[0].tolist()
            
            # Decode the response
            full_response = ''.join([itos[i] for i in out_idx])
            generated_text = full_response[len(prompt):]
            
            return jsonify({'response': generated_text.strip()})
            
        except Exception as e:
            print(f"Error in chat endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"Starting server on port {port}...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Vocab path: {VOCAB_PATH}")
    print(f"Meta path: {META_PATH}")
    app.run(host='0.0.0.0', port=port)
