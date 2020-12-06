import numpy as np
import pandas as pd

# Example from:
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer(smooth_idf=False, norm='l1')
X = vectorizer.fit_transform(corpus)


# My additions to improve understanding
df = pd.DataFrame(X.todense())
df.columns = vectorizer.get_feature_names()
df.index = corpus

# print the result
print("\nTfidfVectorizer() output:\n")
print(df.round(2))


# ----------------------------------------------------------------------------
# manual calculation using CountVectoriser (implemented from scratch in ch7_CountVectorizer.py)

n = len(corpus)
print(f'{n} documents in total.')

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
X = X.todense()

counts_df = pd.DataFrame(X)
words_per_doc_df = counts_df.sum(axis=1)

tf_df = counts_df.div(words_per_doc_df, axis=0)
tf_df.columns = vectorizer.get_feature_names()
tf_df.index = corpus

# print the result
print("\ntf_df manual calculation:")
print(tf_df.round(2))

# inverse frequency calculation
word_presence = (tf_df > 0).astype(float)
print("\nword_presence:\n", word_presence)

# document count (in how many documents does a word occur?)
doc_count = word_presence.sum(0)
print('doc_count:\n', doc_count)

# idf(t) = log [ n / df(t) ] + 1
inv_doc_freq = np.log10(n / doc_count) + 1
print(inv_doc_freq)


tfidf = tf_df * inv_doc_freq
print(tfidf)
