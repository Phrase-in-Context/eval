""" Bert model for Phrase search """

""" HuggingFace API"""
import sys
sys.path.append("..")

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

from .abs_scorer import AbsScorer
from config import CreateLogger
from config import MAX_BATCH_BERT

MAX_BATCH = MAX_BATCH_BERT


class BertScorer(AbsScorer):

    def __init__(self, scorer_type, model_fpath):
        """ """
        self.logger = CreateLogger()
        
        self.logger.debug("[model]: BertPretrainedScorer")
        self.logger.debug("[scorer_type]: %s", scorer_type)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info("cpu/gpu: %s", self.device)

        # THANG
        self.model_name = scorer_type
        # self.model_name = '../../data/pretrained_models/bert-base-uncased/finetuned-11'   # BERT-base-QA
        # self.model_name = '../../data/pretrained_models/bert-large-uncased/finetuned-11'  # BERT-large-QA

        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=self.model_name, config=self.config)

        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.to(self.device)
        super().__init__(self.tokenizer)

    def _embed_batch(self, list_inputText, max_length=64, contextual=False):
        """ """
        self.model.eval()

        with torch.no_grad():

            # First-batch
            rst = self._transformer_embedding_batch(list_inputText[:MAX_BATCH], max_length=max_length, contextual=contextual).cpu()

            # Additional-batch if the size of the list_inputText is larger than MAX_BATCH
            itr_additional = int(len(list_inputText) / MAX_BATCH)
        
            for i in range(itr_additional):
                start_index = (i+1)*MAX_BATCH
                
                list_candidates = list_inputText[start_index:start_index+MAX_BATCH]

                if len(list_candidates) > 0:         
                    rst_tmp = self._transformer_embedding_batch(list_inputText[start_index:start_index+MAX_BATCH], max_length=max_length, contextual=contextual).cpu()
                    rst = torch.cat((rst, rst_tmp), dim=0)

            return rst

    def _transformer_embedding_batch(self, list_inputText, max_length=64, contextual=False):
        list_text = []
        list_attn_mask = []

        for text in list_inputText:

            ret = self.tokenizer.encode_plus(text=text, max_length=max_length, padding='max_length',  truncation=True, add_special_tokens=True)

            # Make batch for pytorch
            list_text.append(ret['input_ids'])
            list_attn_mask.append(ret['attention_mask'])

        list_text = torch.as_tensor(list_text)
        list_attn_mask = torch.as_tensor(list_attn_mask)   # [batch, seq]

        # Assign to gpu (or cpu)
        list_text = list_text.to(self.device)
        list_attn_mask = list_attn_mask.to(self.device)

        outputs = self.model(input_ids=list_text, attention_mask=list_attn_mask, output_hidden_states=True)

        # (one for the output of the embeddings + one for the output of each layer) of shape [batch, seq, dim]
        hidden_states = outputs[2]

        # Average last layer - reasonable choice (sentence-transformer)
        last_layer = hidden_states[-1]                              # [batch, seq, dim]
        unsqueeze_mask = torch.unsqueeze(list_attn_mask, dim=-1)    # [batch, seq, 1]
        masked_last_layer = last_layer * unsqueeze_mask             # [batch, seq, dim]

        # ThangPM
        if contextual:
            return masked_last_layer                                # [batch, seq, dim]

        sum_mask = torch.sum(list_attn_mask, dim=-1, keepdim=True)
        mean_last = torch.sum(masked_last_layer, dim=1) / sum_mask
    
        return mean_last                                            # [batch, dim]


