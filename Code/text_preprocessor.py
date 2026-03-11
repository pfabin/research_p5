import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data --> works kinda like an if statment if you have it already on your machine it will be skipped
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize once to avoid recreating on every review
stop_words = set(stopwords.words("english")) # also converts into set
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    # makes everything lowercase
    text = text.lower()
    # Removes punctuation, numbers and special characters
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenizes by splitting on whitespace
    tokens = text.split()
    # Removes stop words
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize each token
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def preprocess_series(text_series):
    # Applies preprocess_text to every review
    return text_series.apply(preprocess_text)


if __name__ == "__main__":
    from data import load_data

    X_train, X_test, y_train, y_test = load_data(
        "C:/Users/andre/OneDrive/Desktop/Research Project/clean (1).csv"
    )

    X_train_processed = preprocess_series(X_train)
    X_test_processed = preprocess_series(X_test)

    print("Verification: Before vs After Preprocessing")
    for i in range(3):
        print(f"Review {i+1}")
        print(f"BEFORE: {X_train.iloc[i]}")
        print(f"AFTER:  {X_train_processed.iloc[i]}\n")

    # We verify that everything worked correctly and view reviews before and after preprocessing --> can be ignored later