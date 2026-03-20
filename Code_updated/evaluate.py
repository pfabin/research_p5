import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(model, X_test, y_test, model_name="Model"):
    # Get predictions
    y_pred = model.predict(X_test)

    # Print header and accuracy
    print(f"\n=== Evaluation Report: {model_name} ===")
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


# because pytorch models dont have a predict method we have to add a new function
def evaluate_pytorch_model(model, test_loader, y_test, device, model_name="Model"):
    # Switch to evaluation mode - disables dropout (very important for evaluation, dropout should only be used for training)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad(): # we disable gradient tracking for efficiency
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            outputs = model(batch_x)

            predicted = outputs.argmax(dim=1)

            # Move to CPU and convert to numpy for sklearn metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    # Convert to numpy arrays
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    print(f"\n=== Evaluation Report: {model_name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    # Precision, recall and F1 for each class
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Negative", "Neutral", "Positive"]
    ))

    # Confusion matrix with readable labels
    print("Confusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print(f"{'':15} {'Negative':>10} {'Neutral':>10} {'Positive':>10}")
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Neutral", "Positive"]
    for i, row in enumerate(cm):
        print(f"{labels[i]:15} {row[0]:>10} {row[1]:>10} {row[2]:>10}")