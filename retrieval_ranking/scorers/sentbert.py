""" sentence bert (emnlp-19) model for Phrase search """

import sys
sys.path.append("..")

import spacy
import torch
from sentence_transformers import SentenceTransformer

from .abs_scorer import AbsScorer
from config import CreateLogger
from config import MAX_BATCH_SENTENCEBERT


class SentenceBertScorer(AbsScorer):

    def __init__(self, scorer_type, model_fpath):
        """ """
        self.logger = CreateLogger()
        self.logger.debug("[model]: SentenceBertScorer")
        self.logger.debug("[scorer_type]: %s", scorer_type)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info("cpu/gpu: %s", self.device)

        self.nlp = spacy.load("en_core_web_sm")

        self.model_name = scorer_type if not model_fpath else model_fpath
        self.model = SentenceTransformer(self.model_name)
        self.tokenizer = self.model.tokenizer

        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.to(self.device)
        super().__init__(self.tokenizer)

    def embed_batch(self, list_inputText, max_length=64, contextual=False):
        """ """
        self.model.eval()

        with torch.no_grad():

            # First-batch
            rst = self.embedding_batch(list_inputText[:MAX_BATCH_SENTENCEBERT], contextual=contextual)

            # Additional-batch if the size of the list_inputText is larger than MAX_BATCH_SENTENCEBERT
            itr_additional = int(len(list_inputText) / MAX_BATCH_SENTENCEBERT)

            for i in range(itr_additional):
                start_index = (i+1)*MAX_BATCH_SENTENCEBERT
                
                list_candidates = list_inputText[start_index:start_index+MAX_BATCH_SENTENCEBERT]

                if len(list_candidates) > 0:                
                    rst_tmp = self.embedding_batch(list_inputText[start_index:start_index+MAX_BATCH_SENTENCEBERT], contextual=contextual)
                    rst = rst + rst_tmp

            return rst

    def embedding_batch(self, list_inputText, contextual=False):
        list_inputText = [x.lower().strip() for x in list_inputText]

        output_value = "sentence_embedding" if not contextual else "token_embeddings"
        outputs = self.model.encode(list_inputText, convert_to_tensor=True, output_value=output_value, show_progress_bar=False)  # [batch, dim]

        if output_value == "sentence_embedding":
            return outputs.cpu()

        outputs = [output.cpu() for output in outputs]

        return outputs



