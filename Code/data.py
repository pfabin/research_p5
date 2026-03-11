import pandas as pd
from sklearn.model_selection import train_test_split

def map_sentiment(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

    # This function is used to map our rating_review into a 3-class sentiment label


def load_data(filepath, test_size=0.2, random_state=42):

    df = pd.read_csv(filepath)
    df = df[["review_full", "rating_review", "sentiment"]]

    # We use the map_sentiment function to create 3-class sentiment label from rating
    df["sentiment_label"] = df["rating_review"].apply(map_sentiment)

    X = df["review_full"]
    y = df["sentiment_label"]

    # We perform a stratified split to preserve class distribution --> balancing might be done later
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Now we verify that everything worked correctly
    print("First 10 rows of X_train")
    print(X_train.head(10).to_string())

    print("First 10 rows of X_test")
    print(X_test.head(10).to_string())

    print("Class distribution in y_train")
    print(y_train.value_counts().sort_index())

    print("Class distribution in y_test")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("C:/Users/andre/OneDrive/Desktop/Research Project/clean (1).csv")

    # This block is just to verify everything works early on (can be ignored later)