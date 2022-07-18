import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from spanPooling import spanPooling
from densephrases import DensePhrases
from simcse import SimCSE


def load_model(model_path, spanRep=False):

    if model_path == 'bert-base-uncased' or model_path == 'bert-large-uncased':
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    elif 'spanbert' in model_path:
        if spanRep:
            word_embedding_model = models.Transformer(model_path)
            # ThangPM: For no_context cases: Default is mean pooling
            pooling_model = spanPooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            word_embedding_model = models.Transformer(model_path)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    elif 'densephrase' in model_path:
        model = DensePhrases(load_dir=model_path, dump_dir="")
    elif 'simcse' in model_path:
        model = SimCSE(model_path)
    else:
        model = SentenceTransformer(model_path)

    return model
