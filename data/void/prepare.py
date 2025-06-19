import os
import pickle

def prepare_void_data():
    # Get the absolute path to input.txt
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '..', 'input.txt')
    with open(input_path, 'r', encoding='utf-8') as f:
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
