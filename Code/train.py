import time
from data import load_data
from text_preprocessor import preprocess_series
from text_representations import train_word2vec, get_document_vectors, get_bow_vectors
from models import train_svm, train_naive_bayes
from evaluate import evaluate_model


def log_step(message, start=None):
    if start:
        print(f"{message} ({time.time() - start:.1f}s)")
    else:
        print(message)
    return time.time()


def run_pipeline(filepath, models_to_run):

    # Loads data
    t = log_step("Loading data...")
    X_train, X_test, y_train, y_test = load_data(filepath)
    t = log_step("Done!", t)

    # Preprocess text
    t = log_step("Preprocessing text...")
    X_train_processed = preprocess_series(X_train)
    X_test_processed = preprocess_series(X_test)
    t = log_step("Done!", t)

    # Trains Word2Vec (only if needed)
    w2v_model = None
    if any(m in models_to_run for m in ["svm_w2v", "nb_w2v"]):
        t = log_step("Training Word2Vec...")
        w2v_model = train_word2vec(X_train_processed)
        t = log_step("Done!", t)

    # Generatse vectors (only what is needed)
    X_train_w2v, X_test_w2v = None, None
    X_train_bow, X_test_bow = None, None

    if any(m in models_to_run for m in ["svm_w2v", "nb_w2v"]):
        t = log_step("Generating Word2Vec document vectors...")
        X_train_w2v = get_document_vectors(X_train_processed, w2v_model)
        X_test_w2v = get_document_vectors(X_test_processed, w2v_model)
        t = log_step("Done!", t)

    if any(m in models_to_run for m in ["svm_bow", "nb_bow"]):
        t = log_step("Generating BoW vectors...")
        X_train_bow, X_test_bow, _ = get_bow_vectors(X_train_processed, X_test_processed)
        t = log_step("Done!", t)

    # Train and evaluate selected models
    print("Training and evaluating models...")

    if "svm_w2v" in models_to_run:
        t = log_step("Training SVM + Word2Vec...")
        model = train_svm(X_train_w2v, y_train)
        t = log_step("Done!", t)
        evaluate_model(model, X_test_w2v, y_test, model_name="SVM + Word2Vec")

    if "svm_bow" in models_to_run:
        t = log_step("Training SVM + BoW...")
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