import random

import numpy as np
from gensim.models import Doc2Vec
from numpy import dot, concatenate
from numpy.linalg import norm, inv

from code.document import Document
from code.key import SecuredKey
from code.constants import DIMENSIONS


class DataUser:
    def __init__(self):
        # The ranked search with multiple queried keywords
        print('Input query')
        self.query = input().split(' ')
        # The m -dimensional plaintext query feature vector
        self.query_vector = None
        # The m -dimensional trapdoor vector, which is the encrypted form of Vq
        self.secured_query_vector = None

    def get_query_vector(self, key: SecuredKey, model: Doc2Vec):
        """
        generates the trapdoor vector vq_s
        :param model: Doc2Vec model
        :param query: Queried keywords
        :param key: Secured key
        :return: trapdoor vector vq_s = {vq_s′,vq_s″}={VQ′⋅M1^−1,VQ″⋅M2^−1}
        """
        # TODO Normalize
        self.query_vector = model.infer_vector(self.query)
        self.query_vector = self.query_vector / norm(self.query_vector)

        # Split
        vq1 = [None] * DIMENSIONS
        vq2 = [None] * DIMENSIONS
        for j in range(DIMENSIONS):
            if key.S[j] == 0:
                vq1[j] = random.random()
                vq2[j] = self.query_vector[j] - vq1[j]
            if key.S[j] == 1:
                vq1[j] = vq2[j] = self.query_vector[j]

        vq1 = np.array(vq1)
        vq2 = np.array(vq2)

        vq1_s = dot(vq1, inv(key.M1))
        vq2_s = dot(vq2, inv(key.M2))

        vq_s = concatenate((vq1_s, vq2_s))

        return vq_s

    def decrypt(self, documents: [(Document, int)], key: SecuredKey):
        r = []
        for d, s in documents:
            r.append((s, d.decrypt(key)))
        return r
