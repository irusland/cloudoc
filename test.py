import time

from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize


def main():
    model = Doc2Vec.load("d2v.model")
    import numpy as np

    words = "i love kite".split()

    len_before = len(model.docvecs)  # number of docs

    # word vectors for king, queen, man
    w_vec0 = model[words[0]]
    w_vec1 = model[words[1]]
    w_vec2 = model[words[2]]

    new_vec = model.infer_vector(words)

    len_after = len(model.docvecs)

    print(np.array_equal(model[words[0]], w_vec0))
    print(np.array_equal(model[words[1]], w_vec1))
    print(np.array_equal(model[words[2]], w_vec2))

    print(len_before == len_after)



if __name__ == '__main__':
    main()
