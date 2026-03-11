from train import run_pipeline

# Modular Control panel - add or remove models to run
# Available options: "svm_w2v", "svm_bow", "nb_bow", "nb_w2v"
MODELS_TO_RUN = [
    "svm_w2v",
    "svm_bow",
    "nb_bow",
    "nb_w2v",
]

FILEPATH = "C:/Users/andre/OneDrive/Desktop/Research Project/clean (1).csv"

if __name__ == "__main__":
    run_pipeline(FILEPATH, MODELS_TO_RUN)