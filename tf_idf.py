import collections

import pandas as pd
import numpy as np

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]


class tfidf:
    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_len = len(self.corpus)
        self.vocab, self.idf = self.fit()

    def countTf(self, document, word):
        c=0
        for each in document.split(' '):
            if each == word:
                c+=1
        return c

    def fit(self):
        words = [token for each in self.corpus for token in each.split(' ')]
        vocab = collections.Counter(words)
        idf = {}
        for each in vocab.keys():
            c = 0
            for sent in self.corpus:
                if each in sent.split(' '):
                    c += 1
            idf[each] = np.log(self.corpus_len/c+1)
        return vocab, idf

    def predict(self, text):
        embed = []
        for token in text.split(' '):
            tf = self.countTf(text, token)
            if token in self.idf.keys():
                idf = self.idf[token]
            else:
                # self.corpus.append(text)
                # self.vocab, self.idf = self.fit()
                # idf = self.idf[token]
                idf = np.log(self.corpus_len+1/1)  # idf for unknown words either gets ignored or gets added to the corpus and fit again
            embed.append(tf*idf)
        return embed


embedding = tfidf(corpus)
print(embedding.idf)
print(embedding.vocab)
print(embedding.predict('This is an orange'))



