import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    # Print header and accuracy
    print(f"Evaluation Report: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Precision, recall and F1 for each class
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Negative", "Neutral", "Positive"]
    ))

    # Confusion matrix with readable labels
    print("Confusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print(f"{'':15} {'Negative':>10} {'Neutral':>10} {'Positive':>10}")
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Negative", "Neutral", "Positive"]
    for i, row in enumerate(cm):
        print(f"{labels[i]:15} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

# No Verification block needed here since this file only makes sense when called with a trained model