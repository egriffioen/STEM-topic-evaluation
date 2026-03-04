from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

class Doc2VecVectorMaker:
    def __init__(self, documents, vector_size=100):
        tagged_docs = [
            TaggedDocument(simple_preprocess(doc), [i])
            for i, doc in enumerate(documents)
        ]

        self.model = Doc2Vec(
            tagged_docs,
            vector_size=vector_size,
            window=5,
            min_count=2,
            workers=4,
            epochs=40
        )

    def getVector(self, document):
        tokens = simple_preprocess(document)
        return self.model.infer_vector(tokens).tolist()