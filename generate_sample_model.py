import os
import pickle
import torch
from pathlib import Path

def create_sample_model():
    """Create sample model files for development/testing."""
    print("Generating sample model files...")
    
    # Create directories if they don't exist
    os.makedirs('out', exist_ok=True)
    os.makedirs('data/void', exist_ok=True)
    
    # Create a tiny sample model
    vocab_size = 256  # ASCII characters
    block_size = 64
    n_layer = 4
    n_head = 4
    n_embd = 128
    
    # Generate vocabulary
    chars = [chr(i) for i in range(vocab_size)]
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    # Save vocabulary
    with open('data/void/vocab.pkl', 'wb') as f:
        pickle.dump((chars, stoi), f)
    print("✓ Created vocab.pkl")
    
    # Save meta configuration
    meta = {
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'bias': True,
    }
    with open('data/void/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print("✓ Created meta.pkl")
    
    # Create a tiny transformer model
    model = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4*n_embd,
            batch_first=True
        ),
        num_layers=n_layer
    )
    
    # Save model
    torch.save(model.state_dict(), 'out/model.pt')
    print("✓ Created model.pt")
    
    print("\nSample model files generated successfully!")

if __name__ == '__main__':
    create_sample_model()
