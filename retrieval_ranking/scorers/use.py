""" Universal Sentence Encoder (USE) v4 """

""" Tensorflow HUB """

import tensorflow as tf
import os
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from commons import AbsScorer
from project_config import ROOT_DIR
from project_config import MAX_BATCH_USE
from logger import CreateLogger

MAX_BATCH = MAX_BATCH_USE


class USEScorer(AbsScorer):

    def __init__(self, scorer_type="v4", model_fpath="", use_cache=False):
        """ """

        self.use_cache = use_cache
        self.dic_cache = {}

        self.logger = CreateLogger()
        self.logger.debug("[model]: USEScorer")
        self.logger.debug("[scorer_type]: %s", scorer_type)
        self.logger.debug("[model_fpath]: %s", model_fpath)
        self.logger.debug("[batch_size]: %d", MAX_BATCH)
        self.logger.debug("[use_cache]: %s", use_cache)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        if scorer_type == "v4":
            self.logger.info("USE type: %s", scorer_type)

            target_path = os.path.join(ROOT_DIR, "../data/pretrained_models/use-v4/")
            os.environ["TFHUB_CACHE_DIR"] = target_path
            self.logger.info("cache_dir = %s", target_path)

            if os.path.isdir(target_path):
                self.logger.info("[cached] skip downloading the model")
            else:
                self.logger.info("download model (USE-v4)")

                if os.path.isdir(os.path.join(ROOT_DIR, "../data/pretrained_models/")) == False:
                    os.mkdir(os.path.join(ROOT_DIR, "../data/pretrained_models/"))

                os.mkdir(target_path)

            self.model_name = "universal-sentence-encoder-large-v4"
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        elif scorer_type == "v5":
            self.logger.info("USE type: %s", scorer_type)

            target_path = os.path.join(ROOT_DIR, "../data/pretrained_models/use-v5/")
            os.environ["TFHUB_CACHE_DIR"] = target_path
            self.logger.info("cache_dir = %s", target_path)

            if os.path.isdir(target_path):
                self.logger.info("[cached] skip downloading the model")
            else:
                self.logger.info("download model (USE-v5)")

                if os.path.isdir(target_path) == False:
                    os.makedirs(target_path)

            self.model_name = "universal-sentence-encoder-large-v5"
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

        else:
            self.logger.error("not supported type of transformers: %s", scorer_type)

        self.cache = {}

    def _embed_batch(self, list_inputText):
        """ """
        # first-batch
        rst = self.model(list_inputText[:MAX_BATCH])
        # self.logger.debug('GPU batch: 1')

        # additional-batch if the size of the list_inputText is larger than MAX_BATCH_SENTENCEBERT
        itr_additional = int(len(list_inputText) / MAX_BATCH)

        for i in range(itr_additional):
            start_index = (i + 1) * MAX_BATCH
            # self.logger.debug('GPU batch: %d', i+1)

            list_candidates = list_inputText[start_index:start_index + MAX_BATCH]

            if len(list_candidates) > 0:
                rst_tmp = self.model(list_inputText[start_index:start_index + MAX_BATCH])
                rst = np.concatenate((rst, rst_tmp), axis=0)

        return rst


