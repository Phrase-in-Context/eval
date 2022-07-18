""" SimCSE model for Phrase search   """
""" https://arxiv.org/abs/2104.08821 """
""" HuggingFace API"""

import torch
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from commons import AbsScorer
from project_config import MAX_BATCH_SimCSE
from logger import CreateLogger
from project_config import ROOT_DIR
from operator import itemgetter

from transformers import AutoModel, AutoTokenizer
from simcse import SimCSE
import spacy

MAX_BATCH = MAX_BATCH_SimCSE


class SimCSEScorer(AbsScorer):

    def __init__(self, scorer_type):
        """ """
        self.logger = CreateLogger()
        self.cache_dir = os.path.join(ROOT_DIR, "../data/pretrained_models/simcse")
        
        self.logger.debug("[model]: SimCSEScorer")
        self.logger.debug("[scorer_type]: %s", scorer_type)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info("cpu/gpu: %s", self.device)

        self.nlp = spacy.load("en_core_web_sm")

        if scorer_type[:11] == "simcse-self":
            self.logger.error("Not implemented")
#             from transformers import BertConfig, BertTokenizer, BertModel
#             self.logger.info("transformer type: %s", scorer_type)
            
#             config = BertConfig.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir)
#             self.tokenizer   = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=self.cache_dir)
#             self.model = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',
#                                                    config=config,
#                                                    cache_dir=self.cache_dir
#                                                   )

        elif scorer_type == "simcse-roberta":
            self.logger.info("transformer type: %s", scorer_type)

            self.model_name = 'princeton-nlp/sup-simcse-roberta-large'
            self.simcse = SimCSE(self.model_name)

            self.tokenizer = self.simcse.tokenizer
            self.model = self.simcse.model

        elif scorer_type in ["simcse-bert-base-uncased", "simcse-bert-large-uncased", "simcse-roberta-large"]:
            self.logger.info("transformer type: %s", scorer_type)

            # self.model_name = "princeton-nlp/sup-{}".format(scorer_type)
            self.model_name = '../../data/pretrained_models/princeton-nlp/sup-simcse-bert-base-uncased/finetuned-11'

            self.simcse = SimCSE(self.model_name)

            self.tokenizer = self.simcse.tokenizer
            self.model = self.simcse.model

        # freeze model parameters
        # if scorer_type != "simsce-roberta":
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.to(self.device)
        
        self.cache = {}

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

            return rst

    def _transformer_embedding_batch(self, list_inputText, max_length=64, contextual=False, use_cls=False):

        inputs = self.tokenizer(list_inputText, max_length=max_length, padding='max_length', truncation=True, add_special_tokens=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        if contextual:
            return outputs.last_hidden_state

        final_results = []
        for idx, token_embeddings in enumerate(outputs.last_hidden_state):
            att_mask = list(inputs['attention_mask'][idx])
            last_token_index = att_mask.index(0) - 1 if 0 in att_mask else len(att_mask) - 1
            final_results.append(token_embeddings[:last_token_index+1].mean(dim=0))

        return torch.stack(final_results)
