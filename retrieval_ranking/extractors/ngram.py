import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import spacy
from nltk.util import ngrams
from config import CreateLogger
from commons import AbsExtractor


class NGramExtractor(AbsExtractor):
    
    def __init__(self, ngram_min=2, ngram_max=3, remove_stopword=False):
        """ """
        self.logger = CreateLogger()
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        
        self.remove_stopword = remove_stopword
        self.nlp = spacy.load("en_core_web_sm")
        
        self.logger.info('ngrams (%d)-(%d)', self.ngram_min, self.ngram_max)

    def extract(self, text):
        """ """
        phrases = []

        for n in range(self.ngram_min, self.ngram_max+1):
            phrases.extend(self._get_ngrams(text, n))
        
        phrases = [x for x in phrases if len(x) != 0]
        
        return list(phrases)

    def _get_ngrams(self, text, n):
        tokens = [token.text.lower() for token in self.nlp.tokenizer(text)]
        list_ngram = [' '.join(grams).replace(" 's", "'s").replace(" - ", "-").replace(" / ", "/").replace(" %", "%") for grams in ngrams(tokens, n)]

        return list_ngram

    def extract_with_index(self, sent, ngram_min=None, ngram_max=None):
        phrases = []

        if ngram_min and ngram_max:
            for n in range(ngram_min, ngram_max + 1):
                phrases.extend(self._get_ngrams_with_index(sent, n))
        else:
            for n in range(self.ngram_min, self.ngram_max + 1):
                phrases.extend(self._get_ngrams_with_index(sent, n))

        # Filter candidates starting/ending with stop word
        phrases = [x for x in phrases if len(x) != 0]

        return list(phrases)

    def _get_ngrams_with_index(self, text, n):
        list_ngram = []

        tokens = [token.text.lower() for token in self.nlp.tokenizer(text)]
        words_indices = [(word, i) for i, word in enumerate(tokens)]
        ngrams_indices = [grams for grams in ngrams(words_indices, n)]

        for grams_indices in ngrams_indices:
            phrase, indices = [], []
            for gram, idx in grams_indices:
                phrase.append(gram)
                indices.append(idx)

            list_ngram.append((" ".join(phrase), indices))

        return list_ngram

