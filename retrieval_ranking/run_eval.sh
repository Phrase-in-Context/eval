
function run() {
  local DATASET=$1
  local SUBSET=$2
  local EXTRACTOR=$3
  local NGRAMS_MIN=$4
  local NGRAMS_MAX=$5
  local SCORER=$6
  local SCORE_TYPE=$7
  local MAX_SEQ_LENGTH=$8
  local ORACLE=$9
  local CONTEXTUAL=${10}
  local CONTEXT_WINDOW=${11}
  local GPU=${12}

  echo "Start evaluating dataset: '${DATASET}'..."
  echo "Using GPU: ${GPU}"

  export CUDA_VISIBLE_DEVICES=$GPU

  if [[ ${CONTEXTUAL} == "True" ]]; then
    OUTPUT_DIR=../results/"${DATASET}"/"${SUBSET}"/ranking/contextual/"${SCORE_TYPE}"
  else
    OUTPUT_DIR=../results/"${DATASET}"/"${SUBSET}"/ranking/non_contextual/"${SCORE_TYPE}"
  fi

  mkdir -p ${OUTPUT_DIR}

  local params=()

  params+=(--dataset "${DATASET}")
  params+=(--data_subset "${SUBSET}")
  params+=(--extractor "${EXTRACTOR}")
  params+=(--ngram_min "${NGRAMS_MIN}")
  params+=(--ngram_max "${NGRAMS_MAX}")
  params+=(--scorer "${SCORER}")
  params+=(--scorer_type "${SCORE_TYPE}")
  params+=(--max_seq_length "${MAX_SEQ_LENGTH}")
  params+=(--outdir "${OUTPUT_DIR}")

  if [[ ${ORACLE} == "True" ]]; then
    params+=(--oracle_candidates)
  fi

  if [[ ${CONTEXTUAL} == "True" ]]; then
    params+=(--contextual)
    params+=(--context_window "${CONTEXT_WINDOW}")
  fi

  if [[ ${DEBUG} == "True" ]]; then
    params+=(--debug)
  fi

  echo "${params[@]}"
  nohup python eval.py "${params[@]}" > ${OUTPUT_DIR}/eval_logs.txt &
}

dataset=phrase_retrieval
data_subset=PR-pass
extractor=ngrams
max_seq_len=256

oracle=True
contextual=True
DEBUG=True


run ${dataset} "${data_subset}" "${extractor}" 2 3 BERT "bert-base-uncased" "${max_seq_len}" "${oracle}" "${contextual}" -1 0
run ${dataset} "${data_subset}" "${extractor}" 2 3 BERT "bert-large-uncased" "${max_seq_len}" "${oracle}" "${contextual}" -1 1
run ${dataset} "${data_subset}" "${extractor}" 2 3 SentenceBERT "sentence-transformers/bert-base-nli-stsb-mean-tokens" 128 "${oracle}" "${contextual}" -1 2
run ${dataset} "${data_subset}" "${extractor}" 2 3 PhraseBERT "whaleloops/phrase-bert" 128 "${oracle}" "${contextual}" -1 3
run ${dataset} "${data_subset}" "${extractor}" 2 3 SpanBERT "SpanBERT/spanbert-base-cased" "${max_seq_len}" "${oracle}" "${contextual}" -1 4
run ${dataset} "${data_subset}" "${extractor}" 2 3 DensePhrases "princeton-nlp/densephrases-multi-query-multi" "${max_seq_len}" "${oracle}" "${contextual}" -1 5
run ${dataset} "${data_subset}" "${extractor}" 2 3 SimCSE "princeton-nlp/sup-simcse-bert-base-uncased" "${max_seq_len}" "${oracle}" "${contextual}" -1 6
run ${dataset} "${data_subset}" "${extractor}" 2 3 USE "use-v5" "${max_seq_len}" "${oracle}" "${contextual}" -1 7


