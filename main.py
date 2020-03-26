import copy
import math
import os

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from nltk.tokenize import word_tokenize

def train(model, epochs, tagged_data) -> Doc2Vec:
    # model = Doc2Vec(size=dimensions,
    #                 alpha=alpha,
    #                 min_alpha=0.00025,
    #                 min_count=1,
    #                 dm=1)

    model.build_vocab(tagged_data)


    model.train(tagged_data, total_examples=len(tagged_data), epochs=epochs)
    for epoch in range(epochs):
        print(f'iteration {epoch}')
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    return model


def main():
    documents = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Human computer interaction Graph minors A survey human computer",
        "interaction with man",
        "human practice with the computer",
        "kitesurfing is a beautiful sport to practice"
    ]

    tagged_data = [
        TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for
        i, _d in enumerate(documents)]

    filename = "word2vec.model"
    if os.path.exists(filename):
        model = Word2Vec.load(filename)
        print("Loaded from ", filename)
    else:
        dimensions = 5
        model = Doc2Vec(size=dimensions,
                        window=1,
                        min_count=1,
                        workers=8,
                        alpha=0.025,
                        min_alpha=0.01,
                        dm=0)
        model = train(model, 100, tagged_data)
        print("Model trained")

    model.save(filename)
    print("Model Saved ", filename)

    query = 'practice'
    query_list = query.split(' ')
    query_vec = model.infer_vector(query_list)
    sim = model.similar_by_vector(query_vec)

    results = []
    for i in range(len(model.docvecs)):
        docvec = model.docvecs[i]
        similarity = get_cos(docvec, query_vec)
        results.append((i, documents[i], similarity))

    print('for "', ' '.join(query_list), '" by me')
    for i, text, similarity in sorted(results, key=lambda d: -d[2]):
        print(i, text, similarity)

    print()
    print('for "', ' '.join(query_list), '" by gensim')
    sims = model.docvecs.most_similar([query_vec])
    for sid, sim in sims:
        id = int(sid)
        print(id, documents[id], sim)


def get_cos(a, b):
    from numpy import dot
    from numpy.linalg import norm

    return dot(a, b) / (norm(a) * norm(b))

if __name__ == '__main__':
    main()
