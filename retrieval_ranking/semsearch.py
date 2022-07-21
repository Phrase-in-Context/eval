""" """
from commons import AbsSearch
from config import CreateLogger
from config import REMOVE_CANDIDATE_STOPWORDS

from scorers import BertScorer, SentenceBertScorer, SpanBertScorer, USEScorer, SimCSEScorer, DensePhrasesScorer
from extractors import NGramExtractor

import spacy
from spacy.lang.en import English
spacy_sent_splitter = English()
spacy_sent_splitter.add_pipe("sentencizer")
nlp = spacy.load("en_core_web_sm")

import numpy as np

import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')


class SemanticSearch(AbsSearch):

    def __init__(self, use_cache=False):
        self.logger = CreateLogger()
        self.scorer = None
        
        self.logger.info("REMOVE_CANDIDATE_STOPWORDS: %s", REMOVE_CANDIDATE_STOPWORDS)

        self.sentences = []
        self.phrases = list()
        self.contextual = False
        self.list_oracle = []
        self.use_cache = use_cache
        self.dic_cache_candidates = {}

        self.logger.info("REMOVE_CANDIDATE_STOPWORDS: %s", REMOVE_CANDIDATE_STOPWORDS)
        self.logger.debug("[use_cache]: %s", use_cache)

        self.nlp = spacy.load("en_core_web_sm")

    def set_extractor(self, extractor, ngram_min, ngram_max):
        """ """
        if extractor == "ngrams":
            instantiated_extractor = NGramExtractor(ngram_min, ngram_max, REMOVE_CANDIDATE_STOPWORDS)
        else:
            assert False

        self.extractor = instantiated_extractor
        
    def set_scorer(self, scorer, model_fpath="", scorer_type=""):
        """ """
        if scorer == "BERT":
            instantiated_scorer = BertScorer(scorer_type, model_fpath)
        elif scorer in ["SentenceBERT", "PhraseBERT"]:
            instantiated_scorer = SentenceBertScorer(scorer_type, model_fpath)
        elif scorer == "SpanBERT":
            instantiated_scorer = SpanBertScorer(scorer_type, model_fpath)
        elif scorer == "SimCSE":
            instantiated_scorer = SimCSEScorer(scorer_type, model_fpath)
        elif scorer == "DensePhrases":
            instantiated_scorer = DensePhrasesScorer(scorer_type, model_fpath)
        elif scorer == "USE":
            instantiated_scorer = USEScorer(scorer_type)
        else:
            assert False

        self.scorer = instantiated_scorer

    def set_text(self, context, contextual=False, scorer=None, max_seq_length=128):
        sentences = tokenizer.tokenize(context)

        self.sentences.clear()
        self.contextual = contextual
        self.max_seq_length = int(max_seq_length)

        # Some transformers cannot handle long sentences --> Skip those sentences to avoid crashing the process
        if scorer != "USE":
            for sent in sentences:
                tokens = [token.text.lower() for token in self.nlp.tokenizer(sent)]
                encoded_sent = self.scorer.tokenizer.encode_plus(text=' '.join(tokens), add_special_tokens=True)
                encoded_sent = np.array(encoded_sent["input_ids"])[np.array(encoded_sent["attention_mask"]) == 1]
                if len(encoded_sent) >= int(self.max_seq_length):
                    continue

                self.sentences.append(sent)
        else:
            self.sentences = sentences

        if not contextual:
            self._update_phrases()
        else:
            self._update_phrases_with_index()

    def add_oracles(self, set_oracle, gt_sentence, gt_sent_idx):
        if not self.contextual:
            self.list_oracle = list(set_oracle)
            return self.list_oracle
        else:
            self.list_oracle.clear()

            if gt_sent_idx == -1:
                return []

            answer_tokens = [token.text.lower() for token in self.nlp.tokenizer(list(set_oracle)[0])]
            if answer_tokens == ['u.k', '.', 'branch']:
                answer_tokens = ['u.k.', 'branch']
            elif answer_tokens == ['u.s', '.', 'government', 'statistics']:
                answer_tokens = ['u.s.', 'government', 'statistics']
            elif answer_tokens == ['u.s', '.', 'border']:
                answer_tokens = ['u.s.', 'border']
            elif answer_tokens == ['u.s', '.', 'government', 'control']:
                answer_tokens = ['u.s.', 'government', 'control']
            elif answer_tokens == ['u.s', '.', 'mail']:
                answer_tokens = ['u.s.', 'mail']

            for phrase, index in self.extractor.extract_with_index(gt_sentence, ngram_min=2, ngram_max=len(answer_tokens)):
                if ' '.join(answer_tokens) in phrase.strip():
                    self.list_oracle.append((' '.join(answer_tokens), index, gt_sent_idx))

            if len(self.list_oracle) == 0:
                print("FAILED TO ADD ORACLE {} WITH TEXT {}".format(list(set_oracle), gt_sentence))

            return [' '.join(answer_tokens)]

    def _update_phrases(self):
        if self.extractor:
            self.phrases.clear()

            for sentence in self.sentences:
                for phrase in self.extractor.extract(sentence):
                    if phrase.strip().lower() == "":
                        continue
                    else:
                        self.phrases.append(phrase.strip().lower())

            # Remove duplicates in the phrase-candidate list
            self.phrases = list(set(self.phrases))

    def _update_phrases_with_index(self):
        if self.extractor:
            self.phrases.clear()

            for sent_idx, sent in enumerate(self.sentences):
                for phrase, index in self.extractor.extract_with_index(sent):
                    if phrase.strip() == "":
                        continue

                    self.phrases.append((phrase.strip().lower(), index, sent_idx))

    '''
    desc : 1. compute matching scores between query and phrase
         : 2. set_text() should be called before "search"
    query: query to be compared
    top_n: number of results 
    '''
    def search(self, query, top_n=10, window_size=0):
        """ """
        self.logger.debug('number of candidates: %d', len(self.phrases))

        # apply scorer
        if hasattr(self.scorer.__class__, 'score_batch') and callable(getattr(self.scorer.__class__, 'score_batch')):
            phrases = list(self.phrases)
            
            if len(phrases) == 0:
                phrases = ['NA']

            if not self.contextual:
                scores = self.scorer.score_batch(query, phrases, self.list_oracle)

                phrases.extend(self.list_oracle)  # add oracle to the list
                results = [{"phrase": phrase, "score": score} for phrase, score in zip(phrases, scores)]
            else:
                # ThangPM: Score batch for contextual embeddings
                scores = self.scorer.score_batch_contextual(query, phrases, self.sentences, self.list_oracle, self.max_seq_length,
                                                            window_size=window_size, use_context_query=True)

                phrases.extend(self.list_oracle)  # add oracle to the list
                results = [{"phrase": phrase[0], "score": score} for phrase, score in zip(phrases, scores)]
        else:
            results = [{"phrase": phrase, "score": self.scorer.score(query, phrase)} for phrase in self.phrases]

        results = sorted(results, reverse=True, key=lambda x: x['score'])[:top_n]
        
        return results

    '''
    desc       : compute matching scores between query and phrase
    query      : query to be compared
    list_phrase: phrases to be comapred with query
    '''
    def match(self, query, list_phrases):

        self.logger.debug('number of candidates: %d', len(list_phrases))

        # Apply scorer
        if hasattr(self.scorer.__class__, 'score_batch') and callable(getattr(self.scorer.__class__, 'score_batch')):
            scores = self.scorer.score_batch(query, list_phrases)
            results = [{"phrase": phrase, "score": score} for phrase, score in zip(list_phrases, scores)]
        else:
            results = [{"phrase": phrase, "score": self.scorer.score(query, phrase)} for phrase in list_phrases]

        results = sorted(results, reverse=True, key=lambda x: x['score'])

        return results
