import os
import base64
import random

import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# todo change here need retraining model! loaded will contain old M
M = 20


class Corpus:
    FILENAME = "word2vec.model"
    RETRAIN = False
    # Needs to be a large data set
    DOCUMENTS = [
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
        "kitesurfing is a beautiful sport to practice",
        "kite is a flying item",
        "you can also use foil board with kite",
    ]

    def __init__(self):
        if os.path.exists(self.FILENAME) and not self.RETRAIN:
            self.model = Doc2Vec.load(self.FILENAME)
            print("Loaded from ", self.FILENAME)
        else:
            tagged_data = [
                TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
                for i, _d in enumerate(self.DOCUMENTS)]
            self.model = Doc2Vec(size=M,
                                 window=1,
                                 min_count=1,
                                 workers=8,
                                 alpha=0.025,
                                 min_alpha=0.01,
                                 dm=0,
                                 negative=0)  # negative to avoid
                                              # randomisation of infervector
            self._train(100, tagged_data)
            print("Model trained")

        self.model.save(self.FILENAME)
        print("Model Saved ", self.FILENAME)

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
        self.D = [
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
            "kitesurfing is a beautiful sport to practice",
            "kite is a flying item",
            "you can also use foil board with kite",
        ]

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
            self.DV.append(v / np.linalg.norm(v))
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
            # Secure index
            I[i] = np.concatenate(
                (np.dot(np.array(DV1[i]), SK.M1.T),
                 np.dot(np.array(DV2[i]), SK.M2.T)))

            # Encrypt documents
            D̃[i] = SK.encrypt(D[i].encode())

        self.global_DV1 = np.array(DV1)
        self.global_DV2 = np.array(DV2)
        # TODO Outsources I, D̃ to CS
        #      Outsources SK, Model to DU
        return (I, D̃)

    def search(self, Vq, k):
        RList = []
        for i in range(len(self.DV)):
            similarity = get_score(Vq, self.DV[i])
            RList.append((self.D[i], i, similarity))

        return [(d, s) for d, _, s in sorted(RList, key=lambda d: -d[2])][:k]


class DataUser:
    def __init__(self):
        query = 'kite is a sport'

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
        self.Vq = self.Vq / np.linalg.norm(self.Vq)

        # Split
        Vq1 = [None] * M
        Vq2 = [None] * M
        for j in range(M):
            if SK.S[j] == 0:
                Vq1[j] = random.random()
                Vq2[j] = self.Vq[j] - Vq1[j]
            if SK.S[j] == 1:
                Vq1[j] = Vq2[j] = self.Vq[j]

        # Encrypts
        Ṽq1 = np.dot(np.array(Vq1), np.linalg.inv(SK.M1))
        Ṽq2 = np.dot(np.array(Vq2), np.linalg.inv(SK.M2))

        self.global_Vq1 = np.array(Vq1)
        self.global_Vq2 = np.array(Vq2)

        Ṽq = np.concatenate((Ṽq1, Ṽq2))
        return Ṽq

    def decrypt(self, RList: list, SK: SecuredKey):
        r = []
        for d, s in RList:
            r.append((SK.decrypt(d).decode(), s))
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

        return [(d, s) for d, _, s in sorted(RList, key=lambda d: -d[2])][:k]


def main():
    DO = DataOwner()
    DU = DataUser()
    CS = CloudServer()
    k = 3

    SK = DO.GenKey()
    print(SK.S)
    # print(SK.M1, SK.M2)

    DV = DO.DSInfer(DO.D)
    # print(DV)

    I, D̃ = DO.EncData(SK, DV, DO.D)
    # print(I, D̃)

    Ṽq = DU.GenTrapdoor(DU.Q, SK, DO.model)
    # print(Ṽq)

    RListDU = CS.SSearch(D̃, I, Ṽq, k)
    result = DU.decrypt(RListDU, SK)
    print(result)


    Vq = DO.model.infer_vector(DU.Q)
    RListDO = CS.SSearch(D̃, DV, Vq, k)
    result = DU.decrypt(RListDO, SK)
    print(result)

    Rlist_original = DO.search(DO.model.infer_vector(DU.Q), k)
    print(Rlist_original)

    print(DO.model.infer_vector(DU.Q))
    print(DO.model.infer_vector(DU.Q))
    print(DO.model.infer_vector(DU.Q))
    print(DO.model.infer_vector(DU.Q))


def get_score(a, b):
    from numpy import dot
    from numpy.linalg import norm

    return dot(a, b) / (norm(a) * norm(b))


if __name__ == '__main__':
    main()
