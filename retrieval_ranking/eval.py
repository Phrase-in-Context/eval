""" Evaluate the models on Phrase Retrieval dataset
"""

import sys
import os
import time
import json
import random
import argparse
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from datasets import load_dataset

from config import ROOT_DIR
from config import CreateLogger
from semsearch import SemanticSearch


sys.path.append("../../")
logger = CreateLogger()

nltk.download('punkt')
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')


def get_metrics(run_results):
    n_all = 0
    n_top_1, n_top_3, n_top_5, mrr_5 = 0, 0, 0, 0
    n_acc_cand = 0
    n_candidates = 0

    for results in run_results:
        answer = results["ground_truths"][0]
        top_10_candidates = [result["phrase"] for result in results["result"]]

        # Handle this `if` condition in case `contextual` mode is used
        if isinstance(top_10_candidates[0], tuple):
            top_10_candidates = [pred for pred, _, _ in top_10_candidates]

        n_top_1 += 1.0 if answer in top_10_candidates[:1] else 0
        n_top_3 += 1.0 if answer in top_10_candidates[:3] else 0
        n_top_5 += 1.0 if answer in top_10_candidates[:5] else 0

        # ThangPM: pos starts from 1 for MRR
        correct_pred_pos = top_10_candidates[:5].index(answer) + 1 if answer in top_10_candidates[:5] else 0
        mrr_5 += 1.0 / correct_pred_pos if correct_pred_pos > 0 else 0

        n_all += 1
        n_acc_cand += 1 if results['included_in_candidates'] else 0
        n_candidates += results['number_of_candidates']

    if n_all > 0:
        metrics = {
            'Top@1': round(n_top_1 * 100 / n_all, 2),
            'Top@3': round(n_top_3 * 100 / n_all, 2),
            'Top@5': round(n_top_5 * 100 / n_all, 2),
            'MRR@5': round(mrr_5 * 100 / n_all, 2),
            'Avg. acc for candidate extraction': float(n_acc_cand) / n_all,
            'Avg. number of candidates': float(n_candidates) / n_all,
            'Total count': n_all,
        }
    else:
        metrics = {'error': 'n_all is 0'}

    return metrics


def export_results(eval_results, run_results, outdir, contextual):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir, 'eval_results.json'), 'w') as fp:
        json.dump(eval_results, fp, indent=4)
    with open(os.path.join(outdir, 'run_results.json'), 'w') as fp:
        json.dump(run_results, fp, indent=4)

    subdir = "contextual" if contextual else "non_contextual"
    summary_dir = outdir.split(subdir)[0]

    with open(os.path.join(summary_dir, subdir, 'summary.txt'), 'a') as f:
        f.write(outdir + "\t")
        f.write(str(eval_results["Top@1"]) + "\t")
        f.write(str(eval_results["Top@3"]) + "\t")
        f.write(str(eval_results["Top@5"]) + "\t")
        f.write(str(eval_results["MRR@5"]) + "\t")


def extract_context_for_oracle(sentences, answer):
    gt_sentence = ""
    gt_sentence_idx = -1

    for idx, sentence in enumerate(sentences):
        if answer.strip().lower() in sentence.strip().lower():
            gt_sentence = sentence
            gt_sentence_idx = idx

    return gt_sentence, gt_sentence_idx


def run_eval(args, system, examples):
    """ """
    random.seed(42)

    if args.oracle_candidates:
        print("**********************************")
        print("oracle_candidates are included")
        print("**********************************")

    # if "debug" flag is set -> test only 10 samples
    if args.debug:
        examples = examples.select(range(0, 10))

    run_results = []

    for example in tqdm(examples):
        system.set_text(example['context'], contextual=args.contextual, scorer=args.scorer, max_seq_length=args.max_seq_length)
        answers = example['answers']['text']

        # add oracle candidates (spans)
        if args.oracle_candidates:
            logger.debug("ADD ORACLES: %s", answers)
            if args.contextual:
                gt_sentence, gt_sentence_idx = extract_context_for_oracle(system.sentences, answers[0])
                system.add_oracles(set(answers), gt_sentence, gt_sentence_idx)
            else:
                system.add_oracles(set(answers))

        # eval for test queries
        query = example['query'].strip().lower()
        search_result = system.search(query, top_n=10, window_size=int(args.context_window))

        run_results.append({
            'file': example['title'],
            'query': query,
            'ground_truths': example['answers']['text'],
            'number_of_candidates': len(system.phrases + system.list_oracle),
            'included_in_candidates': any(x.strip().lower() in [phrase[0] if system.contextual else phrase
                                          for phrase in system.phrases + system.list_oracle] for x in answers),
            'result': search_result
        })

    eval_results = get_metrics(run_results)

    return eval_results, run_results


def run(args):
    """ """
    # Load data
    data = load_dataset("PiC/" + args.dataset, args.data_subset)["test"]

    # ThangPM: Load system
    system = SemanticSearch()

    # ThangPM: Load model
    model_config = os.path.join(ROOT_DIR, "model_config.json")
    with open(model_config, 'r') as f:
        config = json.load(f)

    model_fpath = [x for x in config if (x['scorer'] == args.scorer and x['scorer_type'] == args.scorer_type)][-1]['model_fpath']
    if model_fpath != "":
        model_fpath = os.path.join(ROOT_DIR, model_fpath)

    system.set_scorer(args.scorer, model_fpath, args.scorer_type)
    system.set_extractor(args.extractor, int(args.ngram_min), int(args.ngram_max))

    # ThangPM: Run evaluation
    eval_results, run_results = run_eval(args, system, data)
    eval_results['dataset'] = args.dataset
    eval_results['data_subset'] = args.data_subset
    eval_results['scorer'] = args.scorer
    eval_results['scorer_type'] = args.scorer_type
    eval_results['extractor'] = args.extractor
    eval_results['oracle'] = args.oracle_candidates
    eval_results['contextual'] = args.contextual

    # Export configs and results
    with open(os.path.join(args.outdir, 'configs.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    export_results(eval_results, run_results, args.outdir, args.contextual)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--extractor', help='model name', choices=['ngrams', 'noun_chunks'])
    parser.add_argument('--ngram_min', type=int, help='ngram min', default=2)
    parser.add_argument('--ngram_max', type=int, help='ngram max', default=3)
    parser.add_argument('--scorer', help='model name', choices=['BERT', 'SentenceBERT', 'SpanBERT', 'USE', 'SimCSE', 'PhraseBERT', 'DensePhrases'])
    parser.add_argument('--scorer_type', help='transformers type',
                        choices=['bert-base-uncased', 'bert-large-uncased', 'sentence-transformers/bert-base-nli-stsb-mean-tokens',
                                 'SpanBERT/spanbert-base-cased', 'use-v5', 'princeton-nlp/sup-simcse-bert-base-uncased',
                                 'whaleloops/phrase-bert', 'princeton-nlp/densephrases-multi-query-multi',
                                 'bert-base-uncased-qa', 'bert-large-uncased-qa', 'sbert-base-nli-stsb-mean-tokens-qa',
                                 'phrase-bert-qa', 'spanbert-base-cased-qa', 'sup-simcse-bert-base-uncased-qa'], default="")

    parser.add_argument('--dataset', default="PiC/phrase_retrieval")
    parser.add_argument('--data_subset', help='subset of the dataset')

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--oracle_candidates', action="store_true")
    parser.add_argument('--contextual', action="store_true")
    parser.add_argument('--context_window', type=int, help='context boundary for a phrase', default=-1)
    parser.add_argument('--max_seq_length', type=int, help='define max seq length of a sentence to handle', default=128)
    parser.add_argument('--outdir', help='output directory')

    args = parser.parse_args()

    format_string = " - {:35}: {}"
    print("{:=^70}".format(" EVALUATION CONFIGURATION "))
    print("Summary")
    print("-" * 70)
    for k, v in vars(args).items():
        print(format_string.format(k, v))

    start_time = time.time()
    sys.argv = [sys.argv[0]]    # Remove arguments to avoid crashing DensePhrases's argparse

    if args.scorer == "USE" and args.contextual:
        sys.exit('Message: USE-v5 is only supported for non-contextual phrase embeddings. Please try other models!')
    elif args.scorer == "DensePhrases":
        sys.exit("Message: DensePhrases is currently not supported. Please try other models!")

    # ThangPM: Run semantic search
    run(args)

    logger.info("elapsed time: %.2f s", time.time() - start_time)


