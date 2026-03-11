from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB

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

# No verification block needed here since we can't meaningfully test the model without evaluation