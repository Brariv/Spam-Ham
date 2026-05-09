import nltk
from nltk.tokenize import word_tokenize

def clean_df(df, stemmer, stop_words):
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    # Tokenization
    df["SMS_TEXT"] = df["SMS_TEXT"].fillna("").astype(str).apply(word_tokenize)

    # Lowercase
    df["SMS_TEXT"] = df["SMS_TEXT"].apply(lambda x: [word.lower() for word in x])

    # Remove punctuation
    df["SMS_TEXT"] = df["SMS_TEXT"].apply(lambda x: [word for word in x if word.isalnum()])

    # Remove stopwords
    df["SMS_TEXT"] = df["SMS_TEXT"].apply(lambda x: [word for word in x if word not in stop_words])

    # Stemming
    df["SMS_TEXT"] = df["SMS_TEXT"].apply(lambda x: [stemmer.stem(word) for word in x])

    # Join tokens back into a string for later vectorization
    df["SMS_TEXT"] = df["SMS_TEXT"].apply(lambda x: ' '.join(x))

    return df