import argparse
import os
import pickle
import numpy as np

def encode(s, stoi):
    return [stoi[c] for c in s]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"length of dataset in characters: {len(data):,}")
    print(f"all the unique characters: \n{''.join(chars)}")
    print(f"vocab size: {vocab_size}")

    stoi = { ch:i for i,ch in enumerate(chars) }
    encode_fn = lambda x: [stoi[c] for c in x]

    data_enc = np.array(encode_fn(data), dtype=np.uint16)

    n = len(data_enc)
    train_data = data_enc[:int(n*0.9)]
    val_data = data_enc[int(n*0.9):]

    out_dir = os.path.join('data', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    train_file = os.path.join(out_dir, 'train.bin')
    val_file = os.path.join(out_dir, 'val.bin')
    vocab_file = os.path.join(out_dir, 'vocab.pkl')
    meta_file = os.path.join(out_dir, 'meta.pkl')

    train_data.tofile(train_file)
    val_data.tofile(val_file)
    with open(vocab_file, 'wb') as f:
        pickle.dump((chars, stoi), f)
    with open(meta_file, 'wb') as f:
        pickle.dump({'vocab_size': vocab_size}, f)

    print(f"train has {len(train_data):,} tokens")
    print(f"val has {len(val_data):,} tokens")

if __name__ == '__main__':
    main()
