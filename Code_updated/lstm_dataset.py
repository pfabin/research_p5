import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

def build_vocab(X_train, min_freq=2):
    counter = Counter()
    for tokens in X_train:
        counter.update(tokens)

    # Start vocab with special tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}

    # Add words that appear at least min_freq times
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab

# We use this function to build up our vocabluary, all words that appear more than once get added.

def tokens_to_indices(tokens, vocab, max_length=200):
    # Convert words to indices, use <UNK> for unknown words
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    # Truncate if too long
    indices = indices[:max_length]

    # Pad if too short
    padding = [vocab["<PAD>"]] * (max_length - len(indices))
    indices = indices + padding

    return indices

# we use this for padding, limiting the review length and to handel unknown words

class ReviewDataset(Dataset):
    def __init__(self, X, y, vocab, max_length=200):
        self.X = [tokens_to_indices(tokens, vocab, max_length) for tokens in X]
        self.y = list(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def get_pretrained_embeddings(vocab, w2v_model):
    # Get embedding dimension from Word2Vec model
    embedding_dim = w2v_model.vector_size

    # Create zero matrix of shape (vocab_size, embedding_dim)
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    found = 0
    for word, idx in vocab.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            found += 1

    print(f"Pretrained embeddings: {found}/{len(vocab)} words found in Word2Vec vocabulary")

    return torch.FloatTensor(embedding_matrix)


def get_dataloaders(X_train, X_test, y_train, y_test, vocab,
                    batch_size=32, max_length=200):
    train_dataset = ReviewDataset(X_train, y_train, vocab, max_length)
    test_dataset = ReviewDataset(X_test, y_test, vocab, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

if __name__ == "__main__":
    from data import load_data
    from text_preprocessor import preprocess_series

    X_train, X_test, y_train, y_test = load_data(
        "C:/Users/andre/OneDrive/Desktop/Research Project/clean (1).csv"
    )

    print("Preprocessing...")
    X_train_processed = preprocess_series(X_train)
    X_test_processed = preprocess_series(X_test)

    print("Building vocabulary...")
    vocab = build_vocab(X_train_processed)

    print("Creating dataloaders...")
    train_loader, test_loader = get_dataloaders(
        X_train_processed, X_test_processed, y_train, y_test, vocab
    )

    # Verification
    print("\n=== Verification ===")
    print(f"Vocabulary size: {len(vocab)}")
    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch input shape: {batch_x.shape}")
    print(f"Batch label shape: {batch_y.shape}")
    print(f"Sample sequence (first 10 indices): {batch_x[0][:10]}")