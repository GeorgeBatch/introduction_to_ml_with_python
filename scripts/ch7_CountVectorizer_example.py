import numpy as np
import pandas as pd

# Example from:
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.shape)


# My additions to improve understanding
df = pd.DataFrame(X.todense())
df.columns = vectorizer.get_feature_names()
df.index = corpus

print(df)

# ----------------------------------------------------------------------------
# manual calculation

# preprocessing
corpus_preprocessed = [doc.lower() for doc in corpus]
corpus_preprocessed = [doc.split() for doc in corpus_preprocessed]
corpus_preprocessed = [[word.strip(',.?!') for word in doc] for doc in corpus_preprocessed]
print('corpus_preprocessed:', corpus_preprocessed)


# initialize where to record
words_to_documentcounts = {}
n = len(corpus_preprocessed)
print(f"total # of documents = {n}")

for i in range(n):
    for word in corpus_preprocessed[i]:
        if word not in words_to_documentcounts:
            words_to_documentcounts[word] = np.zeros(n, dtype=int) # one position per document

        words_to_documentcounts[word][i] += 1

all_words = sorted(list(words_to_documentcounts.keys()))
df_manual = pd.DataFrame(words_to_documentcounts)
df_manual = df[all_words]
df_manual.index = corpus
print(df_manual)


print("\nFidelity check passed:", (df_manual == df).all().all())
