import time

from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import gensim.downloader as api


def main():
    info = api.info()  # show info about available models/datasets
    print(info)
    model = api.load("glove-twitter-25")
    model.most_similar("cat")


if __name__ == '__main__':
    main()
