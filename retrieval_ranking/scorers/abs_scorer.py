import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import spacy


class AbsScorer(object):
    def __init__(self, tokenizer):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = tokenizer

    def embed_batch(self, list_inputText, max_length=64, contextual=False):
        pass

    def score_batch(self, query, list_phrase, list_oracle=None):
        """ """
        list_phrase = list_phrase + (list_oracle if list_oracle else [])

        query_emb = self.embed_batch([query])               # [batch, dim]
        phrase_emb = self.embed_batch(list_phrase)          # [batch, dim]

        score = cosine_similarity(query_emb, phrase_emb)    # [batch]

        return score.tolist()[-1]

    def score_batch_contextual(self, query, list_phrase, sentences, list_oracle=None, max_seq_length=128, window_size=-1, use_context_query=True):
        """ """
        context_phrases, context_queries = [], []
        list_phrase = list_phrase + (list_oracle if list_oracle else [])

        # Prepare contextual phrases and queries for phrase embeddings
        for phrase, index, sent_idx in list_phrase:
            if isinstance(sentences[sent_idx], str):
                tokens = [token.text.lower() for token in self.nlp.tokenizer(sentences[sent_idx])]
            else:
                tokens = sentences[sent_idx]

            if window_size >= 0:
                left_context = index[0] - window_size if index[0] - window_size > 0 else 0
                right_context = index[-1] + window_size + 1 if index[-1] + window_size + 1 < len(tokens) else len(tokens)
            else:
                left_context = 0
                right_context = len(tokens)

            context_phrase = " ".join(tokens[left_context:index[0]]) + " " + phrase + " " + " ".join(tokens[index[-1] + 1:right_context])
            context_phrases.append(context_phrase.strip())

            if use_context_query:
                context_query = " ".join(tokens[left_context:index[0]]) + " " + query + " " + " ".join(tokens[index[-1] + 1:right_context])
                context_queries.append(context_query.strip())

        # Contextual phrase candidates
        context_phrase_embs = self.embed_batch(context_phrases, max_length=max_seq_length, contextual=True)
        phrase_embs = self.extract_contextual_phrase_embeddings_with_context_window([phrase[0] for phrase in list_phrase], context_phrases, context_phrase_embs, max_length=max_seq_length)

        # Contextual queries
        context_query_embs = None
        if use_context_query:
            context_query_embs = self.embed_batch(context_queries, max_length=max_seq_length, contextual=True)
            query_emb = self.extract_contextual_phrase_embeddings_with_context_window([query] * len(list_phrase), context_queries, context_query_embs, max_length=max_seq_length)
            score = cosine_similarity(query_emb, phrase_embs)  # [batch]

            del context_query_embs
            del query_emb

            del context_phrase_embs
            del phrase_embs
            torch.cuda.empty_cache()

            # Extract scores and convert them to float since float32 is not JSON serializable
            score = [float(score[i][i]) for i in range(len(score))]
            return score

        # Non-contextual query
        query_emb = self.embed_batch([query])               # [batch, dim]
        score = cosine_similarity(query_emb, phrase_embs)   # [batch]

        del context_phrase_embs
        del phrase_embs
        torch.cuda.empty_cache()

        return score.tolist()[-1]

    def extract_contextual_phrase_embeddings_with_context_window(self, list_phrase, context_phrases, context_phrase_embs, max_length=256):
        all_phrase_embs = []

        encoded_phrase_list = self.tokenizer.batch_encode_plus(list_phrase, max_length=128, padding='max_length', truncation=True, add_special_tokens=True)
        encoded_context_list = self.tokenizer.batch_encode_plus(context_phrases, max_length=max_length, padding='max_length', truncation=True, add_special_tokens=True)

        for idx, (encoded_phrase, encoded_sent) in enumerate(zip(encoded_phrase_list["input_ids"], encoded_context_list["input_ids"])):
            encoded_phrase = np.array(encoded_phrase)[np.array(encoded_phrase_list["attention_mask"][idx]) == 1]
            encoded_sent = np.array(encoded_sent)[np.array(encoded_context_list["attention_mask"][idx]) == 1]

            start_idx, end_idx = self.find_sub_list(list(encoded_phrase[1:-1]), list(encoded_sent))
            phrase_indices = list(range(start_idx, end_idx + 1, 1))
            phrase_embs = context_phrase_embs[idx][phrase_indices].mean(dim=0)

            if all(np.isnan(phrase_embs.numpy())):
                continue

            all_phrase_embs.append(phrase_embs)

        return torch.stack(all_phrase_embs)

    def find_sub_list(self, sl, l):
        sll = len(sl)

        if sll > 0:
            for ind in (i for i, e in enumerate(l) if e == sl[0]):
                if l[ind:ind + sll] == sl:
                    return ind, ind + sll - 1

        # Phrase is too long, the tokenizer cannot handle it due to its max_seq_length limitation
        # --> return random indices
        return 0, 1

