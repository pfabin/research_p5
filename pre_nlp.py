import pandas as pd
import nltk
nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
nltk.download('wordnet', force=True)
nltk.download('vader_lexicon', force=True)
nltk.download('averaged_perceptron_tagger', force=True)
nltk.download('punkt_tab', force=True)

data = pd.read_csv("/Users/petarfabinger/Desktop/University/AA:: BA/Y3S2P4/Research/data/clean.csv")
df = data.drop(columns = "Unnamed: 0")
print(df.columns)
print(df.head(5))

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [w.lower() for w in tokens if w.isalpha()
                       and w.lower() not in stop_words]

    lemmas = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmas)




df['processed_text'] = df['review_full'].apply(preprocess_text)
print(df[['review_full', 'processed_text']].head(5))

print(df['processed_text'], 20)

print(df["processed_text"][0])

