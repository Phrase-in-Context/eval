""" Main system """
import sys
sys.path.append("../../smart_find_phrase_search")

import spacy
from enum import Enum

from retrieval_ranking import SemanticSearch
from retrieval_ranking import CreateLogger


class PhraseType(Enum):
    DATE = "DATE"
    CARDINAL = "CARDINAL"
    NAME = "NAME"
    GENERAL = "GENERAL"


class System(object):
    def __init__(self):
        self._logger = CreateLogger()
        self.semantic_search = SemanticSearch()

        self.nlp = spacy.load("en_core_web_sm")

    def set_ss_extractor(self, extractor, ngram_min=2, ngram_max=3):
        """ """
        if self.semantic_search:
            self.semantic_search.set_extractor(extractor, ngram_min, ngram_max)

    def set_ss_scorer(self, scorer, model_fpath="", scorer_type=""):
        """ """
        if self.semantic_search:
            self.semantic_search.set_scorer(scorer, model_fpath, scorer_type)

    def set_text(self, text):
        """ """
        for system in [self.semantic_search]:
            if system:
                system.set_text(text)

    def add_oracles(self, set_oracle):
        for system in [self.semantic_search]:
            if system:
                system.add_oracles(set_oracle)

    def search(self, query, top_n=10, filters=None, text_id=""):
        # default filters
        if filters is None:
            filters = {
                'semantic_search': True,
                'acronym': False,
                'date': False,
                'name': False,
                'number': False
            }

        # get query type
        query_type = self._get_type(query)
        self._logger.debug("query type: %s", query_type)

        # search
        result = []
        if filters['semantic_search']:
            result.extend(self.semantic_search.search(query, top_n=top_n, text_id=text_id))

        result.sort(key=lambda x: x['score'], reverse=True)
        return result

    def match(self, query, list_phrase, filters=None):
        # default filters
        if filters is None:
            filters = {
                'semantic_search': True,
                'acronym': False,
                'date': False,
                'name': False,
                'number': False
            }

        # get query type
        query_type = self._get_type(query)
        self._logger.debug("query type: %s", query_type)

        # search
        result = []
        if filters['semantic_search']:
            result.extend(self.semantic_search.match(query, list_phrase))

        result.sort(key=lambda x: x['score'], reverse=True)
        return result

    def _get_type(self, query):
        """ """
        doc = self.nlp(query)

        for ent in doc.ents:
            if ent.label_ == "DATE":
                return PhraseType.DATE

            elif ent.label_ == "CARDINAL":
                return PhraseType.CARDINAL

            elif ent.label_ in ["ORG", "NORP", "LOC", "PERSON"]:
                return PhraseType.NAME

        return PhraseType.GENERAL


