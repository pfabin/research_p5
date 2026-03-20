import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from data import load_data
from text_preprocessor import preprocess_series
from text_representations import train_word2vec, get_document_vectors, get_bow_vectors
from models import train_svm, train_naive_bayes, RNNModel, LSTMModel
from evaluate import evaluate_model, evaluate_pytorch_model
from lstm_dataset import build_vocab, get_dataloaders

# Automatically use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_step(message, start=None):
    if start:
        print(f"{message} ({time.time() - start:.1f}s)")
    else:
        print(message)
    return time.time()


def compute_class_weights(y_train):
    # Compute inverse frequency weights to handle class imbalance -- kinda works "like balanced" allows us to penalize errors on negative / neutal class more
    counts = np.bincount(y_train)
    total = len(y_train)
    weights = total / (len(counts) * counts)
    return torch.FloatTensor(weights).to(device)


def train_pytorch_model(model, train_loader, y_train, epochs=5):
    model = model.to(device)

    class_weights = compute_class_weights(y_train.values) # here we use weights for the mentioned balancing
    criterion = nn.CrossEntropyLoss(weight=class_weights) # use cross entropy as a criterion we used it in deeplearning before...

    # Adam optimizer with standard learning rate --> very strong optimizer we also used in dl
    optimizer = Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad() # resets / clears our gradients

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # We use clipping gradients (very useful)

            optimizer.step()

            # We track our metrics
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1) # takes the class with the highest value/score as prediction
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

    return model


def run_pipeline(filepath, models_to_run):

    # Step 1: Load data
    t = log_step("Step 1: Loading data...")
    X_train, X_test, y_train, y_test = load_data(filepath)
    t = log_step("Done!", t)

    # Step 2: Preprocess text
    t = log_step("\nStep 2: Preprocessing text...")
    X_train_processed = preprocess_series(X_train)
    X_test_processed = preprocess_series(X_test)
    t = log_step("Done!", t)

    # Step 3: Train Word2Vec (only if needed)
    w2v_model = None
    if any(m in models_to_run for m in ["svm_w2v", "nb_w2v"]):
        t = log_step("\nStep 3: Training Word2Vec...")
        w2v_model = train_word2vec(X_train_processed)
        t = log_step("Done!", t)

    # Step 4: Generate vectors (only what is needed)
    X_train_w2v, X_test_w2v = None, None
    X_train_bow, X_test_bow = None, None

    if any(m in models_to_run for m in ["svm_w2v", "nb_w2v"]):
        t = log_step("\nStep 4a: Generating Word2Vec document vectors...")
        X_train_w2v = get_document_vectors(X_train_processed, w2v_model)
        X_test_w2v = get_document_vectors(X_test_processed, w2v_model)
        t = log_step("Done!", t)

    if any(m in models_to_run for m in ["svm_bow", "nb_bow"]):
        t = log_step("\nStep 4b: Generating BoW vectors...")
        X_train_bow, X_test_bow, _ = get_bow_vectors(X_train_processed, X_test_processed)
        t = log_step("Done!", t)

    # Step 5: Build vocab and dataloaders (only if needed)
    if any(m in models_to_run for m in ["rnn", "lstm"]):
        t = log_step("\nStep 4c: Building vocabulary and dataloaders...")
        vocab = build_vocab(X_train_processed)
        train_loader, test_loader = get_dataloaders(
            X_train_processed, X_test_processed, y_train, y_test, vocab
        )
        t = log_step("Done!", t)

    # Step 6: Train and evaluate selected models
    print("\nStep 5: Training and evaluating models...")

    if "svm_w2v" in models_to_run:
        t = log_step("\nTraining SVM + Word2Vec...")
        model = train_svm(X_train_w2v, y_train)
        t = log_step("Done!", t)
        evaluate_model(model, X_test_w2v, y_test, model_name="SVM + Word2Vec")

    if "svm_bow" in models_to_run:
        t = log_step("\nTraining SVM + BoW...")
        model = train_svm(X_train_bow, y_train, max_iter=5000)
        t = log_step("Done!", t)
        evaluate_model(model, X_test_bow, y_test, model_name="SVM + BoW")

    if "nb_bow" in models_to_run:
        t = log_step("\nTraining Naive Bayes + BoW...")
        model = train_naive_bayes(X_train_bow, y_train, representation="bow")
        t = log_step("Done!", t)
        evaluate_model(model, X_test_bow, y_test, model_name="Naive Bayes + BoW")

    if "nb_w2v" in models_to_run:
        t = log_step("\nTraining Naive Bayes + Word2Vec...")
        model = train_naive_bayes(X_train_w2v, y_train, representation="w2v")
        t = log_step("Done!", t)
        evaluate_model(model, X_test_w2v, y_test, model_name="Naive Bayes + Word2Vec")

    if "rnn" in models_to_run:
        t = log_step("\nTraining RNN...")
        model = RNNModel(vocab_size=len(vocab)) #here we define the model RNN/LSTM --> so that train_pytorch can be used for both...
        model = train_pytorch_model(model, train_loader, y_train)
        t = log_step("Done!", t)
        evaluate_pytorch_model(model, test_loader, y_test, device, model_name="RNN") # New evaluate function here !

    if "lstm" in models_to_run:
        t = log_step("\nTraining LSTM...")
        from lstm_dataset import get_pretrained_embeddings
        pretrained_embeddings = get_pretrained_embeddings(vocab, w2v_model)

        model = LSTMModel(
            vocab_size=len(vocab),
            pretrained_embeddings=pretrained_embeddings
        )
        model = train_pytorch_model(model, train_loader, y_train, epochs=8)
        t = log_step("Done!", t)
        evaluate_pytorch_model(model, test_loader, y_test, device, model_name="LSTM")


if __name__ == "__main__":
    run_pipeline(
        "C:/Users/andre/OneDrive/Desktop/Research Project/clean (1).csv",
        ["rnn"]
    )