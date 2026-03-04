from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

class Word2VecVectorMaker:
    def __init__(self, documents, vector_size=100):
        self.tokenized_docs = [simple_preprocess(doc) for doc in documents]

        self.model = Word2Vec(
            sentences=self.tokenized_docs,
            vector_size=vector_size,
            window=5,
            min_count=2,
            workers=4
        )

        self.vector_size = vector_size

    def getVector(self, document):
        tokens = simple_preprocess(document)

        word_vectors = [
            self.model.wv[word]
            for word in tokens
            if word in self.model.wv
        ]

        if not word_vectors:
            return np.zeros(self.vector_size).tolist()

        return np.mean(word_vectors, axis=0).tolist()