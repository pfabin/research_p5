import torch
import torch.nn as nn
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB

## quick code explanation for RNN and LSTM model for my own understanding: embedding layer converts our integer inputs into dense vectors. hidden size determes the dimentionallity of the hidden state (compresseion bottleneck)
## num layers is the number of LSTM / RNN layers used (can be stacked). We use dropout after the final hidden state as a regularization techinques, ensures that the model doesnt rely on a single neuron to havily --> this is automatically disabeled during evaluation


def train_naive_bayes(X_train, y_train, representation="bow"):
    # MultinomialNB for BoW (count based), GaussianNB for Word2Vec (continuous)
    if representation == "bow":
        model = MultinomialNB()
    else:
        model = GaussianNB()

    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, max_iter=1000):
    # Linear SVM with balanced class weights to handle class imbalance
    svm = LinearSVC(class_weight="balanced", max_iter=max_iter)

    # Wrap with calibration to enable probability estimates --> instead of getting a hard perdiction like 1 or 0 we get a probability for each class / sentiment
    calibrated_svm = CalibratedClassifierCV(svm)
    calibrated_svm.fit(X_train, y_train)

    return calibrated_svm

# add a text run function to test everything works correctly before performing hours of training.

def test_run(model, device, vocab_size, seq_len=200, num_classes=3):
    print(f"\n=== Test Run: {model.__class__.__name__} ===")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    model.train()
    model = model.to(device)

    # Create random batch of 4 reviews each of length seq_len
    x = torch.randint(0, vocab_size, (4, seq_len), device=device)
    # Create random labels (0, 1, or 2 for our 3 classes)
    y = torch.randint(0, num_classes, (4,), device=device)

    # Run forward pass
    logits = model(x)

    # Check output shape is (batch_size, num_classes)
    assert logits.shape == (4, num_classes), \
        f"Expected (4, {num_classes}), got {tuple(logits.shape)}"
    print(f"Output shape check passed: {tuple(logits.shape)} ✓")

    # Loss check
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    print(f"Loss computation check passed: loss={loss.item():.4f} ✓")

    # Backward pass check
    loss.backward()

    # Check all parameters received gradients
    missing = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            missing.append(name)

    if missing:
        raise RuntimeError(f"Missing gradients for: {missing}")
    print(f"Gradient flow check passed: all parameters received gradients ✓")

    model.zero_grad(set_to_none=True)
    print(f"Test run completed successfully!\n")


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128,
                 num_layers=2, num_classes=3, dropout=0.5):
        super(RNNModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # only applies dropout if we have more than one layer
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)  # maps our hidden dimensions into our 3 class output / scores
        self._init_weights()

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)
        # embedded: (batch_size, seq_length, embedding_dim)

        output, hidden = self.rnn(embedded)
        # hidden: (num_layers, batch_size, hidden_size)

        # Take final layer's last hidden state
        last_hidden = hidden[-1]
        # last_hidden: (batch_size, hidden_size)

        out = self.dropout(last_hidden)
        out = self.fc(out)
        # out: (batch_size, num_classes)

        return out

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        self.embedding.weight.data[0] = torch.zeros(self.embedding.embedding_dim)

        # used to enicialize our embedding weights

# For the lstm we rempeat the same code and apply only minimal changes (nn.LSTM instead of RNN) and we also have a long term cell state instead of just a hidden state.

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200, hidden_size=256,
                 num_layers=2, num_classes=3, dropout=0.5, pretrained_embeddings=None):
        super(LSTMModel, self).__init__()


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # added code for pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
            print("Pretrained embeddings loaded into embedding layer ✓")

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self._init_weights()

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)
        # embedded: (batch_size, seq_length, embedding_dim)

        output, (hidden, cell) = self.lstm(embedded)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: long term memory, not needed for classification

        last_hidden = hidden[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)

        return out

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        self.embedding.weight.data[0] = torch.zeros(self.embedding.embedding_dim)

# No verification block needed here since we can't meaningfully test the model without evaluation