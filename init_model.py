import os
import pickle
import torch
from model import GPT, GPTConfig

def init_model_files():
    """Initialize sample model files for development/testing."""
    # Create directories if they don't exist
    os.makedirs("out", exist_ok=True)
    os.makedirs("data/void", exist_ok=True)

    # Create a minimal vocabulary
    chars = [chr(i) for i in range(32, 127)]  # Basic ASCII characters
    stoi = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    # Save vocabulary
    with open("data/void/vocab.pkl", "wb") as f:
        pickle.dump((chars, stoi), f)

    # Create meta configuration
    meta = {
        "block_size": 64,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "bias": True,
        "vocab_size": vocab_size
    }

    # Save meta configuration
    with open("data/void/meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # Initialize a small model
    config = GPTConfig(**meta)
    model = GPT(config)

    # Save model
    torch.save(model.state_dict(), "out/model.pt")

    print("Created sample model files:")
    print("- out/model.pt")
    print("- data/void/vocab.pkl")
    print("- data/void/meta.pkl")

if __name__ == "__main__":
    init_model_files()
