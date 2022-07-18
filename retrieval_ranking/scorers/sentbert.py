""" sentence bert (emnlp-19) model for Phrase search """

import torch
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from commons import AbsScorer
from sentence_transformers import SentenceTransformer
from project_config import MAX_BATCH_SENTENCEBERT
from project_config import ROOT_DIR
from logger import CreateLogger
from operator import itemgetter
import spacy


class SentenceBertScorer(AbsScorer):

    def __init__(self, scorer_type, model_fpath):
        """ """
        self.logger = CreateLogger()
        self.logger.debug("[model]: SentenceBertScorer")
        self.logger.debug("[scorer_type]: %s", scorer_type)

        self.cache = {}
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info("cpu/gpu: %s", self.device)

        self.nlp = spacy.load("en_core_web_sm")

        # pretrained models
        # https://www.sbert.net/docs/pretrained_models.html
        # https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
        
        # Semantic Textual Similarity
        if scorer_type == "Sbert-base-nli-stsb-mean-tokens":     # SentenceBERT
            # self.model_name = 'sentence-transformers/bert-base-nli-stsb-mean-tokens'
            self.model_name = '../../data/pretrained_models/sentence-transformers/bert-base-nli-stsb-mean-tokens/finetuned-11'
            self.model = SentenceTransformer(self.model_name)
            self.tokenizer = self.model.tokenizer
                
        elif scorer_type == "Sroberta-base-nli-stsb-mean-tokens":
            self.model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        
        # Duplicate Questions Detection
        elif scorer_type == "Sdistilbert-base-nli-stsb-quora-ranking":
            self.model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')
        
        # Information Retrieval
        elif scorer_type == "Sdistilroberta-base-msmarco-v2":
            
            if os.path.isdir( os.path.join(ROOT_DIR, "../data/pretrained_models/msmarco-distilroberta-base-v2/") ):
                self.logger.info("[cached] skip downloading the model")
                
            else:
                if os.path.isdir( os.path.join(ROOT_DIR, "../data/pretrained_models/") ) == False:
                    os.mkdir( os.path.join(ROOT_DIR, "../data/pretrained_models/") )
                    
                os.system('wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/msmarco-distilroberta-base-v2.zip ' + '--directory-prefix=' + os.path.join(ROOT_DIR, "../data/pretrained_models/"))
                os.system("unzip " + os.path.join(ROOT_DIR, "../data/pretrained_models/msmarco-distilroberta-base-v2.zip") + " -d " + os.path.join(ROOT_DIR, "../data/pretrained_models/msmarco-distilroberta-base-v2"))

            self.model = SentenceTransformer(os.path.join(ROOT_DIR, "../data/pretrained_models/msmarco-distilroberta-base-v2/"))

        # PhraseBERT model
        elif scorer_type == "phrasebert":

            # self.model_name = "../../../phrase2vec/models/model_1_3_epochs_lr_2e-5_bs_2048_sentencebert/checkpoints/300"
            # self.model_name = "../../../phrase2vec/models/model_1_3_epochs_lr_2e-5_bs_2048_sentencebert_nli_stsb_data_v1.2/checkpoints/300"
            # self.model_name = '../../../phrase2vec/models/phrase-bert-model/pooled_context_para_triples_p=0.8'

            # self.model_name = '../../../phrase-bert-topic-model/results/models/PhraseBERT-PR-scratch'
            # self.model_name = '../../../phrase-bert-topic-model/results/models/PhraseBERT-PR'

            # self.model_name = "whaleloops/phrase-bert"                                            # PhraseBERT-HF
            self.model_name = '../../data/pretrained_models/whaleloops/phrase-bert/finetuned-11'    # PhraseBERT-QA
            self.model = SentenceTransformer(self.model_name)
            self.tokenizer = self.model.tokenizer

        # Model 1: Finetune sentenceBert on PPDB
        elif scorer_type == "sentbert-ppdb":
            self.model_name = "../../../phrase2vec/models/model_1_3_epochs_lr_2e-5_bs_2048_sentbert"
            self.model = SentenceTransformer(self.model_name)
            self.tokenizer = self.model.tokenizer
        else:
            self.logger.error("not supported type of transformers: %s", scorer_type)

        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.to(self.device)

    def _embed_batch(self, list_inputText, max_length=64, contextual=False, use_cls=False):
        """ """
        self.model.eval()

        with torch.no_grad():

            # first-batch
            rst = self._embedding_batch(list_inputText[:MAX_BATCH_SENTENCEBERT], max_length=max_length, contextual=contextual, use_cls=use_cls)

            # additional-batch if the size of the list_inputText is larger than MAX_BATCH_SENTENCEBERT
            itr_additional = int(len(list_inputText) / MAX_BATCH_SENTENCEBERT)

            for i in range(itr_additional):
                start_index = (i+1)*MAX_BATCH_SENTENCEBERT
                
                list_candidates = list_inputText[start_index:start_index+MAX_BATCH_SENTENCEBERT]

                if len(list_candidates) > 0:                
                    rst_tmp = self._embedding_batch(list_inputText[start_index:start_index+MAX_BATCH_SENTENCEBERT], max_length=max_length, contextual=contextual, use_cls=use_cls)
                    # rst = torch.cat((rst, rst_tmp), dim=0)
                    rst = rst + rst_tmp

            return rst

    def _embedding_batch(self, list_inputText, max_length=64, contextual=False, use_cls=False):

        list_inputText = [x.lower().strip() for x in list_inputText]

        output_value = "sentence_embedding" if not contextual else "token_embeddings"
        outputs = self.model.encode(list_inputText, convert_to_tensor=True, output_value=output_value, show_progress_bar=False)  # [batch, dim]

        if output_value == "sentence_embedding":
            return outputs.cpu()

        outputs = [output.cpu() for output in outputs]

        return outputs



