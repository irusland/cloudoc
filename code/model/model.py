import logging
import os

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from code.constants import DIMENSIONS
from code.logger.logger import Logger
from definitions import MODEL_PATH, DOCS_DIR


class Corpus:

    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = Doc2Vec.load(MODEL_PATH)
            Logger.info(f'Loaded from {MODEL_PATH}')
        else:
            Logger.exception(f'No model found {MODEL_PATH}')

    @staticmethod
    def train():
        Logger.info(f'Training started')
        documents = []
        for address, dirs, files in os.walk(DOCS_DIR):
            for file in files:
                path = os.path.join(address, file)
                Logger.info(f'Processing {path}')
                with open(path) as f:
                    txt = f.read()
                    documents.append(
                        TaggedDocument(
                            words=gensim.utils.simple_preprocess(txt),
                            tags=[file]))

        model = Doc2Vec(documents,
                        size=DIMENSIONS,
                        window=10,
                        min_count=2,
                        workers=8)

        model.train(documents,
                    total_examples=model.corpus_count,
                    epochs=21)
        Logger.info("Model trained")

        model.save(MODEL_PATH)
        Logger.info(f'Model Saved {MODEL_PATH}')


if __name__ == '__main__':
    logging.basicConfig()
    Corpus.train()
