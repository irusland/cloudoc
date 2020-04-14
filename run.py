import copy

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def practice():
    documents = [
        "human with the calculating machine",
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
        "human with the computer",
    ]

    prep_docs = []
    for i in range(len(documents)):
        doc = documents[i]
        prep_docs.append(TaggedDocument(doc.split(' '), [f'{i}']))
    docs = prep_docs

    # initialize a model
    dimensions = 5
    model = Doc2Vec(size=dimensions, window=1, min_count=1, workers=8,
                    alpha=0.025,
                    min_alpha=0.01, dm=0, negative=0)

    # build vocabulary
    model.build_vocab(docs)

    # get the initial document vector, and most similar articles
    # (before training, the results should be wrong)

    simid = 0

    docvec1 = copy.copy(model.docvecs[0])
    # docvec1 = model.docvecs[0]
    docvecsyn1 = copy.copy(model.docvecs.doctag_syn0[0])
    docsim1 = copy.copy(model.docvecs).most_similar(simid)

    # train this model
    model.train(docs, total_examples=len(docs), epochs=100)

    # get the trained document vector, and most similar articles
    # (after training, the results should be correct)
    docvec2 = model.docvecs[0]
    docvecsyn2 = model.docvecs.doctag_syn0[0]
    docsim2 = model.docvecs.most_similar(simid)

    # print results

    # document vector
    print('Document vector:')

    # before training
    print('(Before training)')
    print(docvec1[:5])
    print(docvecsyn1[:5])

    # we can see that, the document vectors do not change after the training.
    print('(After training, exactly the same.)')
    print(docvec2[:5])
    print(docvecsyn2[:5])

    # most similar documents
    print('\nMost similar:')

    # before training, the result is wrong. after training, correct. good.
    print('(Before training)')
    print(docsim1[:2])

    print('(After training, significantly changed)')
    print(docsim2[:2])

    print(documents[simid])
    for id, delta in docsim2:
        id = int(id)
        print(documents[id], delta)
    for dsyn in docvecsyn2:
        print(dsyn)
        # id = int(id)
        # print(documents[id], delta)

practice()