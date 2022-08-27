
function run {
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

  if [[ ${SCORER} == "USE" ]]; then
    # ThangPM: To fix the issue: "TensorFlow libdevice not found"
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=$(find /usr/ -type d -name nvvm 2>/dev/null)/../
  fi

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

  echo "${params[@]}"
  nohup python eval.py "${params[@]}" > ${OUTPUT_DIR}/eval_logs.txt &
}

function evaluate_model {
  local DATASET=$1
  local DATASET_CONFIG=$2
  local MODEL=$3
  local CONTEXTUAL=$4

  local EXTRACTOR=ngrams
  local MAX_SEQ_LENGTH=256
  local OUTPUT_DIR=../results
  local ORACLE=True

  if [[ ${MODEL} == "BERT-base" ]]; then
    run ${DATASET} "${DATASET_CONFIG}" "${EXTRACTOR}" 2 3 BERT "bert-base-uncased" 256 "${ORACLE}" "${CONTEXTUAL}" -1 0
  elif [[ ${MODEL} == "BERT-large" ]]; then
    run ${DATASET} "${DATASET_CONFIG}" "${EXTRACTOR}" 2 3 BERT "bert-large-uncased" 256 "${ORACLE}" "${CONTEXTUAL}" -1 0
  elif [[ ${MODEL} == "PhraseBERT" ]]; then
    run ${DATASET} "${DATASET_CONFIG}" "${EXTRACTOR}" 2 3 PhraseBERT "whaleloops/phrase-bert" 128 "${ORACLE}" "${CONTEXTUAL}" -1 0
  elif [[ ${MODEL} == "SpanBERT" ]]; then
    run ${DATASET} "${DATASET_CONFIG}" "${EXTRACTOR}" 2 3 SpanBERT "SpanBERT/spanbert-base-cased" 256 "${ORACLE}" "${CONTEXTUAL}" -1 0
  elif [[ ${MODEL} == "SentenceBERT" ]]; then
    run ${DATASET} "${DATASET_CONFIG}" "${EXTRACTOR}" 2 3 SentenceBERT "sentence-transformers/bert-base-nli-stsb-mean-tokens" 128 "${ORACLE}" "${CONTEXTUAL}" -1 0
  elif [[ ${MODEL} == "SimCSE" ]]; then
    run ${DATASET} "${DATASET_CONFIG}" "${EXTRACTOR}" 2 3 SimCSE "princeton-nlp/sup-simcse-bert-base-uncased" 256 "${ORACLE}" "${CONTEXTUAL}" -1 0
  elif [[ ${MODEL} == "USE" ]]; then
    run ${DATASET} "${DATASET_CONFIG}" "${EXTRACTOR}" 2 3 USE "use-v5" 256 "${ORACLE}" "${CONTEXTUAL}" -1 0
  fi
}

# ThangPM: DensePhrase will be supported later
#run ${DATASET} "${DATASET_CONFIG}" "${EXTRACTOR}" 2 3 DensePhrases "princeton-nlp/densephrases-multi-query-multi" 256 "${ORACLE}" "${CONTEXTUAL}" -1 0

"$@"
