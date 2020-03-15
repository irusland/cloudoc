r"""
Similarity Queries
==================

Demonstrates querying a corpus for similar documents.
"""

import logging
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from collections import defaultdict
from gensim import corpora

documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Human computer interaction Graph minors A survey human computer "
    "interaction",
    "human with the computer",
]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]


from gensim import models
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

doc = "measurement of user relation"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
print(vec_lsi)

from gensim import similarities
index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it

index.save('/tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')

sims = index[vec_lsi]  # perform a similarity query against the corpus
print(doc)
print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples

time.sleep(0.1)
sims = sorted(enumerate(sims), key=lambda item: -item[1])
for i, s in sims:
    print(i, s, documents[i])
