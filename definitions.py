import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(ROOT_DIR, 'docs')
CODE_DIR = os.path.join(ROOT_DIR, 'code')
MODEL_DIR = os.path.join(CODE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'word2vec.model')

DEBUG_LOG = os.path.join(ROOT_DIR, 'run.log')
