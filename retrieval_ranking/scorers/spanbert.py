""" span bert (tacl-20) model for Phrase search """

import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from commons import AbsScorer
from project_config import MAX_BATCH_SPANBERT
from logger import CreateLogger
from operator import itemgetter
import spacy


class SpanBertScorer(AbsScorer):

    def __init__(self, scorer_type, model_fpath):
        """ """
        self.logger = CreateLogger()
        self.logger.debug("[model]: SpanBertScorer")
        self.logger.debug("[scorer_type]: %s", scorer_type)
        self.logger.debug("[model_fpath]: %s", model_fpath)
        
        self.cache = {}
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info("cpu/gpu: %s", self.device)

        self.nlp = spacy.load("en_core_web_sm")
        
        if scorer_type == 'span-bert-base-cased':
            from transformers import BertConfig, BertTokenizer, BertModel
            from transformers import AutoTokenizer, AutoModel
            self.logger.info("transformer type: %s", scorer_type)

            # self.model_name = "SpanBERT/spanbert-base-cased"
            self.model_name = '../../data/pretrained_models/SpanBERT/spanbert-base-cased/finetuned-11'
            config = BertConfig.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
            self.model = AutoModel.from_pretrained(self.model_name, config=config)
                
        elif scorer_type == 'span-bert-large-cased':
            from transformers import BertConfig, BertTokenizer, BertModel
            self.logger.info("transformer type: %s", scorer_type)
        
            if os.path.exists(model_fpath) == False:
                self.logger.error("need to download pretrained model of spanbert")
                self.logger.error("PATH: %s", model_fpath)
                raise ValueError("model does not exist")

            # span bert use the same configuration of BERT
            config = BertConfig.from_pretrained(model_fpath, local_files_only=True)   
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
            self.model = BertModel.from_pretrained(model_fpath,
                                                   local_files_only=True,
                                                   config=config)
                
        else:
            self.logger.error("not supported type of transformers: %s", scorer_type)
            self.logger.error("supported type: span-bert-base-cased, span-bert-large-cased")

        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.to(self.device)
        
        self.cache = {}

    def _embed_batch(self, list_inputText, max_length=64, contextual=False, use_cls=False):
        """ """
        self.model.eval()

        with torch.no_grad():

            # first-batch
            rst = self._transformer_embedding_batch(list_inputText[:MAX_BATCH_SPANBERT], max_length=max_length, contextual=contextual, use_cls=use_cls).cpu()

            # additional-batch if the size of the list_inputText is larger than MAX_BATCH_SENTENCEBERT
            itr_additional = int(len(list_inputText) / MAX_BATCH_SPANBERT)
        
            for i in range(itr_additional):
                start_index = (i+1)*MAX_BATCH_SPANBERT
                
                list_candidates = list_inputText[start_index:start_index+MAX_BATCH_SPANBERT]

                if len(list_candidates) > 0:         
                    rst_tmp = self._transformer_embedding_batch(list_inputText[start_index:start_index+MAX_BATCH_SPANBERT], max_length=max_length, contextual=contextual, use_cls=use_cls).cpu()
                    rst = torch.cat((rst, rst_tmp), dim=0)

            return rst

    def _transformer_embedding_batch(self, list_inputText, max_length=64, contextual=False, use_cls=False):

        list_text = []
        list_attn_mask = []

        for text in list_inputText:

            ret = self.tokenizer.encode_plus(text=text, max_length=max_length, padding='max_length',
                                             truncation=True, add_special_tokens=True)

            # make batch for pytorch
            list_text.append(ret['input_ids'])
            list_attn_mask.append(ret['attention_mask'])

        list_text = torch.as_tensor(list_text)
        list_attn_mask = torch.as_tensor(list_attn_mask)   # [batch, seq]

        # assign to gpu (or cpu)
        list_text = list_text.to(self.device)
        list_attn_mask = list_attn_mask.to(self.device)

        outputs = self.model(input_ids=list_text,
                             attention_mask=list_attn_mask,
                             output_hidden_states=True
                             )

        last_hidden_states = outputs[0]  # [batch, seq, dim]
        pooler_output = outputs[1]  # [batch, dim] : special token - linear -> not recommended
        hidden_states = outputs[2]  # (one for the output of the embeddings + one for the output of each layer)
        # of shape [batch, seq, dim]

        # 3. average last two layers
        last_layer = hidden_states[-1]       # [batch, seq, dim]
        last_layer_2nd = hidden_states[-2]   # [batch, seq, dim]

        unsqueeze_mask = torch.unsqueeze(list_attn_mask, dim=-1) # [batch, seq, 1]
        masked_last_layer     = last_layer * unsqueeze_mask      # [batch, seq, dim]
        masked_last_layer_2nd = last_layer_2nd * unsqueeze_mask  # [batch, seq, dim]

        # ThangPM
        if contextual:
            # return last_hidden_states
            # avg_last_2 = (torch.add(masked_last_layer, masked_last_layer_2nd)) / 2
            return masked_last_layer       # [batch, seq, dim]      # <======= USE LAST HIDDEN LAYER
        elif use_cls:
            return pooler_output

        add_last_2 = (torch.add(torch.sum(masked_last_layer, dim=1), torch.sum(masked_last_layer_2nd, dim=1)))/2
        sum_mask = torch.sum(list_attn_mask, dim=-1, keepdim=True)

        mean_last = torch.sum(masked_last_layer, dim=1) / sum_mask
        mean_last_2 = add_last_2 / sum_mask
        final_results = mean_last   # <======= USE MEAN LAST LAYER

        return final_results    # [batch, dim]
