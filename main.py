import base64
import logging
import os
import random

import gensim
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import dot, concatenate
from numpy.linalg import inv, norm

from definitions import TXT_DIR

# todo change here need retraining model! loaded will contain old M
M = 10


class Corpus:
    FILENAME = "word2vec.model"
    RETRAIN = False
    SAVE = True

    def __init__(self):
        logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                            datefmt='%H:%M:%S', level=logging.INFO)

        if os.path.exists(self.FILENAME) and not self.RETRAIN:
            self.model = Doc2Vec.load(self.FILENAME)
            print("Loaded from ", self.FILENAME)
        else:
            documents = []
            for address, dirs, files in os.walk(TXT_DIR):
                for file in files:
                    path = os.path.join(address, file)
                    print(f'Processing {path}')
                    with open(path) as f:
                        txt = f.read()
                        # documents.append(gensim.utils.simple_preprocess(txt))
                        documents.append(
                            TaggedDocument(
                                words=gensim.utils.simple_preprocess(txt),
                                tags=[file]))

            self.model = Doc2Vec(documents,
                                 size=M,
                                 window=10,
                                 min_count=2,
                                 workers=8)  # negative=0 to avoid
                                               # randomisation of infervector

            # self.model.build_vocab(documents)
            self.model.train(documents, total_examples=self.model.corpus_count,
                             epochs=21)
            # self._train(10, documents)
            print("Model trained")

        if self.SAVE:
            self.model.save(self.FILENAME)
            print("Model Saved ", self.FILENAME)
        else:
            print("Model NOT Saved")

    def _train(self, epochs, tagged_data):
        self.model.build_vocab(tagged_data)

        self.model.train(tagged_data, total_examples=len(tagged_data),
                         epochs=epochs)
        for epoch in range(epochs):
            print(f'iteration {epoch}')
            self.model.train(tagged_data,
                             total_examples=self.model.corpus_count,
                             epochs=self.model.iter)
            # decrease the learning rate
            self.model.alpha -= 0.0002
            # fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha


class SecuredKey:
    """
    SK is only shared by DU but protected from CS
    """

    def __init__(self):
        # randomly generated m-bit vector
        self.S: list = None

        # randomly generated m×m-dimensional invertible matrices
        self.M1: np.array = None
        self.M2: np.array = None

        # key for document set encryption
        self.g = None

    def decrypt(self, d̃: bytes) -> bytes:
        f = Fernet(self.g)
        return f.decrypt(d̃)

    def encrypt(self, d: bytes) -> bytes:
        f = Fernet(self.g)
        return f.encrypt(d)


class DataOwner:
    def __init__(self):
        # The document set of n documents
        self.D = []
        for address, dirs, files in os.walk(TXT_DIR):
            for file in files:
                path = os.path.join(address, file)
                with open(path) as f:
                    txt = f.read()
                    self.D.append(txt)

        # The encrypted document set
        self.D̃ = None

        # The dimension vector generated by the Doc2Vec model
        self.m = M

        # Document vectors
        self.DV = []

        # Encrypted form of self.document_vectors
        self.I = None

        self._container = Corpus()
        self.model: Doc2Vec = self._container.model

        self._secret = "irusland"

    def DSInfer(self, D) -> list:
        """
        get m -dimensional feature vector DVi for each document di
        :param D: Documents
        :return: Document vectors
        """
        # TODO is ?!normalized!? and treated as the plaintext index

        for d in D:
            v = self.model.infer_vector(d.split(' '))
            self.DV.append(v / norm(v))
        return self.DV

    def GenSVec(self):
        arr = np.zeros(self.m)
        arr[:random.randrange(self.m)] = 1
        np.random.shuffle(arr)
        return arr

    def GenKey(self) -> SecuredKey:
        """generates the secured key SK = {S, M1, M2, g}"""
        sk = SecuredKey()

        sk.S = self.GenSVec()

        sk.M1 = np.random.randn(self.m, self.m)
        sk.M2 = np.random.randn(self.m, self.m)

        salt = b'\xc8,J\xe2*\xf6\x95\xfa\xf0\x86!\xbe\x0f!]\xf3'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        sk.g = base64.urlsafe_b64encode(kdf.derive(self._secret.encode()))
        return sk

    def EncData(self, SK: SecuredKey, DV: list, D: list):
        """
        encrypt DV and D by SK and generates the secure index I
        and encrypted documents D̃
        :param SK: Secure key
        :param DV: Document Vectors
        :param D: Documents
        :return: (I, D̃) where I={DV′⋅M1^T, DV″⋅M2^T}
        """
        DV1 = [None] * len(self.D)
        DV2 = [None] * len(self.D)
        D̃ = [None] * len(self.D)
        I = [None] * len(self.D)
        for i in range(len(self.D)):
            # Separate vectors
            DV1[i] = [.0] * M
            DV2[i] = [.0] * M

            for vp in range(self.m):
                if SK.S[vp] == 1:
                    DV1[i][vp] = random.random()
                    DV2[i][vp] = DV[i][vp] - DV1[i][vp]
                if SK.S[vp] == 0:
                    DV1[i][vp] = DV[i][vp]
                    DV2[i][vp] = DV[i][vp]

            DV1[i] = np.array(DV1[i])
            DV2[i] = np.array(DV2[i])

            DV1S = dot(DV1[i], SK.M1.T)
            DV2S = dot(DV2[i], SK.M2.T)
            # Secure index
            I[i] = concatenate((DV1S, DV2S))

            # Encrypt documents
            D̃[i] = SK.encrypt(D[i].encode())

        # TODO Outsources I, D̃ to CS
        #      Outsources SK, Model to DU
        return (I, D̃)


class DataUser:
    def __init__(self):
        query = 'bird'

        # The ranked search with multiple queried keywords
        self.Q = query.split(' ')

        # The m -dimensional plaintext query feature vector
        self.Vq = None

        # The m -dimensional trapdoor vector, which is the encrypted form of Vq
        self.Ṽq = None

    def GenTrapdoor(self, Q: list, SK: SecuredKey, model: Doc2Vec):
        """
        generates the trapdoor vector Ṽq
        :param Q: Queried keywords
        :param SK: Secured key
        :return: trapdoor vector Ṽq = {Ṽq′,Ṽq″}={VQ′⋅M1^−1,VQ″⋅M2^−1}
        """
        # TODO Normalize
        self.Vq = model.infer_vector(self.Q)
        self.Vq = self.Vq / norm(self.Vq)

        # Split
        Vq1 = [None] * M
        Vq2 = [None] * M
        for j in range(M):
            if SK.S[j] == 0:
                Vq1[j] = random.random()
                Vq2[j] = self.Vq[j] - Vq1[j]
            if SK.S[j] == 1:
                Vq1[j] = Vq2[j] = self.Vq[j]

        Vq1 = np.array(Vq1)
        Vq2 = np.array(Vq2)

        Vq1S = dot(Vq1, inv(SK.M1))
        Vq2S = dot(Vq2, inv(SK.M2))

        Ṽq = concatenate((Vq1S, Vq2S))

        return Ṽq

    def decrypt(self, RList: list, SK: SecuredKey):
        r = []
        for d, s in RList:
            r.append((s, SK.decrypt(d).decode()))
        return r


class CloudServer:
    def __init__(self):
        self.I = None
        self.D̃ = None

    def SSearch(self, D̃, I, Ṽq, k):
        """
        CS conducts the secure inner product operation between every
        document index in I and the trapdoor Ṽq to perform the
        semantic-aware ranked search.
        :param D̃: Encrypted documents
        :param I: Secure Index
        :param Ṽq: Trapdoor query vector
        :param k: Number of most related documents
        :return: k most related Encrypted documents
        """
        RList = []
        for i in range(len(I)):
            similarity = get_score(Ṽq, I[i])
            RList.append((D̃[i], i, similarity))

        return [(d, s)
                for d, _, s in
                sorted(RList, key=lambda d: -d[2])][:k if k else len(D̃)]


def main():
    DO = DataOwner()
    DU = DataUser()
    CS = CloudServer()
    k = 3
    print(f'Query "{" ".join(DU.Q)}"')
    SK = DO.GenKey()
    print(*SK.S)
    # print(SK.M1, SK.M2)

    DV = DO.DSInfer(DO.D)
    print(*DO.model.infer_vector(DU.Q))

    I, D̃ = DO.EncData(SK, DV, DO.D)
    # print(I, D̃)

    Ṽq = DU.GenTrapdoor(DU.Q, SK, DO.model)
    # print(Ṽq)

    RListDU = CS.SSearch(D̃, I, Ṽq, k)
    result = DU.decrypt(RListDU, SK)
    print("From server ", *result, sep='\n\t')

    Rlist_original = CS.SSearch(D̃, DV, DU.Vq, k)
    result = DU.decrypt(Rlist_original, SK)
    print("From  owner ", *result, sep='\n\t')

    # for d in DO.model.wv.vocab:
    #     print(d)

    print(DO.model.wv.most_similar(positive=['bird']))


def get_score(a, b):
    return dot(a, b)


if __name__ == '__main__':
    main()
