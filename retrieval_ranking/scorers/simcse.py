""" SimCSE model for Phrase search   """
""" https://arxiv.org/abs/2104.08821 """

import sys
sys.path.append("..")

import spacy
import torch
from simcse import SimCSE

from .abs_scorer import AbsScorer
from config import CreateLogger
from config import MAX_BATCH_SimCSE


class SimCSEScorer(AbsScorer):

    def __init__(self, scorer_type, model_fpath):
        """ """
        self.logger = CreateLogger()
        self.logger.debug("[model]: SimCSEScorer")
        self.logger.debug("[scorer_type]: %s", scorer_type)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info("cpu/gpu: %s", self.device)

        self.nlp = spacy.load("en_core_web_sm")

        self.model_name = scorer_type if not model_fpath else model_fpath
        self.simcse = SimCSE(self.model_name)
        self.tokenizer = self.simcse.tokenizer
        self.model = self.simcse.model

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.to(self.device)
        super().__init__(self.tokenizer)

    def embed_batch(self, list_inputText, max_length=64, contextual=False):
        """ """
        self.model.eval()

        with torch.no_grad():

            # First-batch
            rst = self._transformer_embedding_batch(list_inputText[:MAX_BATCH_SimCSE], max_length=max_length, contextual=contextual).cpu()

            # Additional-batch if the size of the list_inputText is larger than MAX_BATCH_SimCSE
            itr_additional = int(len(list_inputText) / MAX_BATCH_SimCSE)
        
            for i in range(itr_additional):
                start_index = (i+1)*MAX_BATCH_SimCSE
                
                list_candidates = list_inputText[start_index:start_index+MAX_BATCH_SimCSE]

                if len(list_candidates) > 0:         
                    rst_tmp = self._transformer_embedding_batch(list_inputText[start_index:start_index+MAX_BATCH_SimCSE], max_length=max_length, contextual=contextual).cpu()
                    rst = torch.cat((rst, rst_tmp), dim=0)

            return rst

    def _transformer_embedding_batch(self, list_inputText, max_length=64, contextual=False):
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
