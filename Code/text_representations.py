import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer


def train_word2vec(tokens, vector_size=100, window=5, min_count=2, workers=8):
    # Trains Word2Vec
    model = Word2Vec(
        sentences=tokens,
        vector_size=vector_size,  # deterimines vector size e.g. 100-dimensional vector
        window=window,  # the amount of words looked at as context 5 to each side
        min_count=min_count,  # used to ignore words that apprar only once
        workers=workers  # uses more cpu cores to speed up training -- IMPORTANT -- turn this off / to zero if you are on Laptop
    )
    # values were suggested by claude --> might be worth to experiment with later on.
    # U need gensim to run this & doesnt work on 3.14 set to 3.11
    return model

def get_document_vectors(tokens, w2v_model):
    # Convert each review to a single vector by averaging its word vectors
    vectors = []
    for review in tokens:
        word_vectors = [
            w2v_model.wv[word] for word in review if word in w2v_model.wv
        ]
        if word_vectors:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            # Fallback to zero vector if no words found in vocabulary --> otherwise it could crash so dont remove this
            vectors.append(np.zeros(w2v_model.vector_size))
    return np.array(vectors)


def get_bow_vectors(X_train_tokens, X_test_tokens):
    # Join tokens back to strings as CountVectorizer expects raw text --> otherwise it wont work
    X_train_strings = [" ".join(tokens) for tokens in X_train_tokens]
    X_test_strings = [" ".join(tokens) for tokens in X_test_tokens]
    vectorizer = CountVectorizer(max_features=10000) # reducing vocab to 10k to speed up training (has only minimal impact on accuracy)
    # Fit only used on train data to avoid leaking test information
    X_train_bow = vectorizer.fit_transform(X_train_strings)
    X_test_bow = vectorizer.transform(X_test_strings)

    return X_train_bow, X_test_bow, vectorizer


if __name__ == "__main__":
    from data import load_data
    from text_preprocessor import preprocess_series

    X_train, X_test, y_train, y_test = load_data(
        "C:/Users/andre/OneDrive/Desktop/Research Project/clean (1).csv"
    )

    X_train_processed = preprocess_series(X_train)
    X_test_processed = preprocess_series(X_test)

    w2v_model = train_word2vec(X_train_processed)

    X_train_vectors = get_document_vectors(X_train_processed, w2v_model)
    X_test_vectors = get_document_vectors(X_test_processed, w2v_model)

    print("Verification")
    print(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
    print(f"X_train_vectors shape: {X_train_vectors.shape}")
    print(f"X_test_vectors shape:  {X_test_vectors.shape}")
    print(f"Sample vector (first review, first 5 dimensions): {X_train_vectors[0][:5]}")

    # Again we verify that everything worked correctly before moving on --> can be ignored later