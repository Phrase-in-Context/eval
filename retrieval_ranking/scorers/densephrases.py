""" DensePhrases model for Phrase search   """
""" https://arxiv.org/abs/2012.12624 """
""" HuggingFace API"""

import torch
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from commons import AbsScorer
from project_config import MAX_BATCH_DENSEPHRASES
from logger import CreateLogger
from project_config import ROOT_DIR

from transformers import AutoModel, AutoTokenizer
from densephrases import DensePhrases
from densephrases.utils.single_utils import load_encoder
from operator import itemgetter
import spacy

MAX_BATCH = MAX_BATCH_DENSEPHRASES


class DensePhrasesScorer(AbsScorer):

    def __init__(self, scorer_type):
        """ """
        self.logger = CreateLogger()
        self.cache_dir = os.path.join(ROOT_DIR, "../data/pretrained_models/densephrases")
        
        self.logger.debug("[model]: DensePhrasesScorer")
        self.logger.debug("[scorer_type]: %s", scorer_type)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info("cpu/gpu: %s", self.device)

        self.nlp = spacy.load("en_core_web_sm")

        if scorer_type == "densephrases-multi-query-multi":
            self.logger.info("transformer type: %s", scorer_type)

            self.model_name = 'princeton-nlp/densephrases-multi-query-multi'
            self.densephrases = DensePhrases(load_dir=self.model_name, dump_dir="")
            self.tokenizer = self.densephrases.tokenizer
            self.model = self.densephrases.model

        if scorer_type == "densephrases-multi":
            self.logger.info("transformer type: %s", scorer_type)

            self.model_name = 'princeton-nlp/densephrases-multi'
            self.densephrases = DensePhrases(load_dir=self.model_name, dump_dir="")
            self.tokenizer = self.densephrases.tokenizer
            self.model = self.densephrases.model

        else:
            self.logger.error("not supported type of transformers: %s", scorer_type)

        # freeze model parameters
        # if scorer_type != "simsce-roberta":
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.to(self.device)
        
        self.cache = {}

    def reset(self):
        self.cache.clear()

    def _embed_batch(self, list_inputText, max_length=64, contextual=False, use_cls=False):
        """ """
        self.model.eval()

        with torch.no_grad():

            # first-batch
            rst = self._transformer_embedding_batch(list_inputText[:MAX_BATCH], max_length=max_length, contextual=contextual, use_cls=use_cls).cpu()

            # additional-batch if the size of the list_inputText is larger than MAX_BATCH_SENTENCEBERT
            itr_additional = int(len(list_inputText) / MAX_BATCH)
        
            for i in range(itr_additional):
                start_index = (i+1)*MAX_BATCH
                
                list_candidates = list_inputText[start_index:start_index+MAX_BATCH]

                if len(list_candidates) > 0:
                    rst_tmp = self._transformer_embedding_batch(list_inputText[start_index:start_index+MAX_BATCH], max_length=max_length, contextual=contextual, use_cls=use_cls).cpu()
                    rst = torch.cat((rst, rst_tmp), dim=0)
                    # rst = np.concatenate((rst, rst_tmp), axis=0)

            return rst

    def _transformer_embedding_batch(self, list_inputText, max_length=64, contextual=False, use_cls=False):

        inputs = self.tokenizer(list_inputText, truncation=True, return_tensors="pt", max_length=max_length, padding='max_length')
        inputs = inputs.to(self.device)
        
        embeddings = self.model.embed_phrase(**inputs)[0]

        if contextual:
            return embeddings

        final_results = []
        for idx, token_embeddings in enumerate(embeddings):
            att_mask = list(inputs['attention_mask'][idx])
            last_token_index = att_mask.index(0) - 1 if 0 in att_mask else len(att_mask) - 1
            final_results.append(token_embeddings[:last_token_index+1].mean(dim=0))                                 # Mean pooling GPU6
            # final_results.append(torch.concat([token_embeddings[0], token_embeddings[last_token_index]]))           # First + last pooling [CLS] + [SEP] GPU5
            # final_results.append(torch.concat([token_embeddings[1], token_embeddings[last_token_index - 1]]))     # First + last pooling excluding [CLS] and [SEP]
            
        return torch.stack(final_results)


