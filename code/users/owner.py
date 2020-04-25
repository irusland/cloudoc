import os
import random

import numpy as np
from gensim.models import Doc2Vec
from numpy import dot, concatenate
from numpy.linalg import norm

from code.key import SecuredKey
from code.constants import DIMENSIONS
from code.model.container import Corpus
from definitions import DOCS_DIR


class DataOwner:
    def __init__(self):
        # The document set of n documents
        self.documents = []
        for address, dirs, files in os.walk(DOCS_DIR):
            for file in files:
                path = os.path.join(address, file)
                with open(path) as f:
                    txt = f.read()
                    self.documents.append(txt)

        # The encrypted document set
        self.secured_documents = None

        # Document vectors
        self.document_vectors = []

        # Encrypted form of self.document_vectors
        self.index = None

        self._container = Corpus()
        self.model: Doc2Vec = self._container.model

    def get_vectors(self) -> list:
        """
        get m -dimensional feature vector DVi for each document di
        :param documents: Documents
        :return: Document vectors
        """

        for d in self.documents:
            v = self.model.infer_vector(d.split(' '))
            self.document_vectors.append(v / norm(v))
        return self.document_vectors

    def encrypt_data(self, key: SecuredKey, doc_vectors: list):
        """
        encrypt DV and D by SK and generates the secure index index
        and encrypted documents d_s
        :param key: Secure key
        :param doc_vectors: Document Vectors
        :return: (index, d_s) where index={DV′⋅M1^T, DV″⋅M2^T}
        """
        dv1 = [None] * len(self.documents)
        dv2 = [None] * len(self.documents)
        d_s = [None] * len(self.documents)
        index = [None] * len(self.documents)
        for i in range(len(self.documents)):
            # Separate vectors
            dv1[i] = [.0] * DIMENSIONS
            dv2[i] = [.0] * DIMENSIONS

            for vp in range(DIMENSIONS):
                if key.S[vp] == 1:
                    dv1[i][vp] = random.random()
                    dv2[i][vp] = doc_vectors[i][vp] - dv1[i][vp]
                if key.S[vp] == 0:
                    dv1[i][vp] = doc_vectors[i][vp]
                    dv2[i][vp] = doc_vectors[i][vp]

            dv1[i] = np.array(dv1[i])
            dv2[i] = np.array(dv2[i])

            dv1_s = dot(dv1[i], key.M1.T)
            dv2_s = dot(dv2[i], key.M2.T)
            # Secure index
            index[i] = concatenate((dv1_s, dv2_s))

            # Encrypt documents
            d_s[i] = key.encrypt(self.documents[i].encode())

        # TODO Outsources index, d_s to CS
        #      Outsources SK, Model to DU
        return index, d_s
