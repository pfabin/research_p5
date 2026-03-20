import torch
from train import run_pipeline
from models import RNNModel, LSTMModel, test_run

# Modular Control panel - add or remove models to run
# Available options: "svm_w2v", "svm_bow", "nb_bow", "nb_w2v", "rnn", "lstm"
MODELS_TO_RUN = [
    # "svm_w2v",
    # "svm_bow",
    # "nb_bow",
    "nb_w2v", # we need at least one w2v model before lstm to ensure pretrained embeddings work
    "lstm",
    # "rnn",
]

FILEPATH = "C:/Users/andre/OneDrive\Desktop/Research Project/clean (1).csv"

VOCAB_SIZE = 53986

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run test checks for any PyTorch models in the list before training
    if "rnn" in MODELS_TO_RUN:
        test_run(RNNModel(vocab_size=VOCAB_SIZE), device, VOCAB_SIZE)

    if "lstm" in MODELS_TO_RUN:
        test_run(LSTMModel(vocab_size=VOCAB_SIZE), device, VOCAB_SIZE)

    run_pipeline(FILEPATH, MODELS_TO_RUN)