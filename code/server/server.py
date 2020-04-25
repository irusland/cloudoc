from code.calculations import score


class CloudServer:
    def __init__(self):
        self.index = None
        self.documents = None

    def search(self, documents, index, query_vector, count):
        """
        CS conducts the secure inner product operation between every
        document index in I and the trapdoor Ṽq to perform the
        semantic-aware ranked search.
        :param documents: Encrypted documents
        :param index: Secure Index
        :param query_vector: Trapdoor query vector
        :param count: Number of most related documents
        :return: k most related Encrypted documents
        """
        result = []
        for i in range(len(index)):
            similarity = score(query_vector, index[i])
            result.append((documents[i], i, similarity))

        return [(d, s)
                for d, _, s in
                sorted(result, key=lambda d: -d[2])][:count if count else len(documents)]
