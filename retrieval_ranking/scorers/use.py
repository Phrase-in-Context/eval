""" Universal Sentence Encoder (USE) v5 """
""" Tensorflow HUB """

import sys
sys.path.append("..")

import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .abs_scorer import AbsScorer
from config import CreateLogger
from config import MAX_BATCH_USE
from config import ROOT_DIR


class USEScorer(AbsScorer):

    def __init__(self, scorer_type="use-v5", model_fpath="", use_cache=False):
        """ """
        self.logger = CreateLogger()
        self.logger.debug("[model]: USEScorer")
        self.logger.debug("[scorer_type]: %s", scorer_type)
        self.logger.debug("[model_fpath]: %s", model_fpath)
        self.logger.debug("[batch_size]: %d", MAX_BATCH_USE)
        self.logger.debug("[use_cache]: %s", use_cache)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.logger.info("USE type: %s", scorer_type)

        target_path = os.path.join(ROOT_DIR, "../../models/use-v5/")
        os.environ["TFHUB_CACHE_DIR"] = target_path
        self.logger.info("Cache_dir = %s", target_path)

        if os.path.isdir(target_path):
            self.logger.info("[cached] Skip downloading the model")
        else:
            self.logger.info("Download model (USE-v5)")

            if os.path.isdir(target_path) == False:
                os.makedirs(target_path)

        self.model_name = "universal-sentence-encoder-large-v5"
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def embed_batch(self, list_inputText, max_length=64, contextual=False):
        """ """
        # First-batch
        rst = self.model(list_inputText[:MAX_BATCH_USE])

        # Additional-batch if the size of the list_inputText is larger than MAX_BATCH_USE
        itr_additional = int(len(list_inputText) / MAX_BATCH_USE)

        for i in range(itr_additional):
            start_index = (i + 1) * MAX_BATCH_USE
            list_candidates = list_inputText[start_index:start_index + MAX_BATCH_USE]

            if len(list_candidates) > 0:
                rst_tmp = self.model(list_inputText[start_index:start_index + MAX_BATCH_USE])
                rst = np.concatenate((rst, rst_tmp), axis=0)

        return rst


