import time

from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize


def main():
    model = Doc2Vec.load("d2v.model")

    tokens = "kite".split()

    new_vector = model.infer_vector(tokens)
    similar_docs = model.docvecs.most_similar([new_vector])
    print('similar ', similar_docs)
    print('similar ', sorted(similar_docs, key=lambda item: -(item[1])))


if __name__ == '__main__':
    main()
