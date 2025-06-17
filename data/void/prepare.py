import os
import pickle

def prepare_void_data():
    # Read the input text
    with open('../input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Get unique characters from the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create character to integer mapping
    stoi = { ch:i for i,ch in enumerate(chars) }
    
    # Save the vocabulary
    with open('vocab.pkl', 'wb') as f:
        pickle.dump((chars, stoi), f)
    
    # Save meta information
    meta = {
        'vocab_size': vocab_size,
        'block_size': 64,  # context length for training
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.1,
    }
    
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Prepared {len(text)} characters of text")
    print(f"Vocabulary size: {vocab_size}")
    print("Files saved: vocab.pkl, meta.pkl")

if __name__ == '__main__':
    prepare_void_data()
