import os

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from code.constants import DIMENSIONS
from code.logger.logger import Logger
from definitions import MODEL_PATH, DOCS_DIR


class Corpus:
    RETRAIN = False
    SAVE = True

    def __init__(self):
        if os.path.exists(MODEL_PATH) and not self.RETRAIN:
            self.model = Doc2Vec.load(MODEL_PATH)
            Logger.info(f'Loaded from {MODEL_PATH}')
        else:
            documents = []
            for address, dirs, files in os.walk(DOCS_DIR):
                for file in files:
                    path = os.path.join(address, file)
                    Logger.info(f'Processing {path}')
                    with open(path) as f:
                        txt = f.read()
                        # documents.append(gensim.utils.simple_preprocess(txt))
                        documents.append(
                            TaggedDocument(
                                words=gensim.utils.simple_preprocess(txt),
                                tags=[file]))

            self.model = Doc2Vec(documents,
                                 size=DIMENSIONS,
                                 window=10,
                                 min_count=2,
                                 workers=8)

            self.model.train(documents, total_examples=self.model.corpus_count,
                             epochs=21)
            Logger.info("Model trained")

        if self.SAVE:
            self.model.save(MODEL_PATH)
            Logger.info(f'Model Saved {MODEL_PATH}')
        else:
            Logger.info('Model NOT Saved')

    def _train(self, epochs, tagged_data):
        self.model.build_vocab(tagged_data)

        self.model.train(tagged_data, total_examples=len(tagged_data),
                         epochs=epochs)
        for epoch in range(epochs):
            Logger.info(f'iteration {epoch}')
            self.model.train(tagged_data,
                             total_examples=self.model.corpus_count,
                             epochs=self.model.iter)
            # decrease the learning rate
            self.model.alpha -= 0.0002
            # fix the learning rate, no decay
            self.model.min_alpha = self.model.alpha