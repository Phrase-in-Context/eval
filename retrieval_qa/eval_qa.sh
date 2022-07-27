
function eval_qa() {

  local MODEL_BASE=$1
  local DATASET=$2
  local DATASET_CONFIG=$3
  local OUTPUT_DIR=$4
  local EVAL_BATCH_SIZE=$5
  local SEED=$6
  local MAX_SEQ_LENGTH=$7
  local GPU=$8

  local MODEL=${OUTPUT_DIR}/${DATASET}/${DATASET_CONFIG}/qa/${MODEL_BASE}/finetuned
  local OUTPUT_DIR=${OUTPUT_DIR}/${DATASET}/${DATASET_CONFIG}/qa/${MODEL_BASE}/evaluation

  mkdir -p $OUTPUT_DIR
  export CUDA_VISIBLE_DEVICES=$GPU

  params=()

  params+=(--model_name_or_path "${MODEL}")
  params+=(--do_predict)
  params+=(--dataset_name PiC/"${DATASET}")
  params+=(--dataset_config_name "${DATASET_CONFIG}")
  params+=(--per_device_eval_batch_size "${EVAL_BATCH_SIZE}")
  params+=(--max_seq_length "${MAX_SEQ_LENGTH}")
  params+=(--doc_stride 128)
  params+=(--output_dir "${OUTPUT_DIR}")
  params+=(--overwrite_output_dir)
  params+=(--overwrite_cache)
  params+=(--seed "${SEED}")

  params+=(--report_to wandb)
  params+=(--run_name "${DATASET}"/"${DATASET_CONFIG}"/qa/"${MODEL_BASE}"/evaluation)

  echo "${params[@]}"
  nohup python -u run_qa.py "${params[@]}" > $OUTPUT_DIR/predict_logs.txt &
}

function evaluate_all() {
  echo "*** EVALUATION STARTED ***"

  local DATASET="phrase_retrieval"   # ["phrase_retrieval", "phrase_sense_disambiguation"]
  local DATASET_CONFIG="PR-pass"     # "PR-pass" OR "PR-page" for "phrase_retrieval" else ""
  local OUTPUT_DIR=../results
  local RANDOM_SEED=42

  eval_qa bert-base-uncased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 0
  eval_qa bert-large-uncased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 1
  eval_qa allenai/longformer-base-4096 "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 2 "${RANDOM_SEED}" 4096 2
  eval_qa allenai/longformer-large-4096 "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 1 "${RANDOM_SEED}" 4096 3
  eval_qa sentence-transformers/bert-base-nli-stsb-mean-tokens "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 4
  eval_qa whaleloops/phrase-bert "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 5
  eval_qa SpanBERT/spanbert-base-cased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 6
  eval_qa princeton-nlp/sup-simcse-bert-base-uncased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 7
}

function evaluate_model() {
  local DATASET=$1
  local DATASET_CONFIG=$2
  local MODEL=$3

  local OUTPUT_DIR=../results
  local RANDOM_SEED=42

  if [[ ${MODEL} == "BERT-base" ]]; then
    eval_qa bert-base-uncased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 0
  elif [[ ${MODEL} == "BERT-large" ]]; then
    eval_qa bert-large-uncased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 0
  elif [[ ${MODEL} == "PhraseBERT" ]]; then
    eval_qa whaleloops/phrase-bert "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 0
  elif [[ ${MODEL} == "SpanBERT" ]]; then
    eval_qa SpanBERT/spanbert-base-cased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 0
  elif [[ ${MODEL} == "SentenceBERT" ]]; then
    eval_qa sentence-transformers/bert-base-nli-stsb-mean-tokens "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 0
  elif [[ ${MODEL} == "SimCSE" ]]; then
    eval_qa princeton-nlp/sup-simcse-bert-base-uncased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 "${RANDOM_SEED}" 512 0
  elif [[ ${MODEL} == "Longformer-base" ]]; then
    eval_qa allenai/longformer-base-4096 "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 2 "${RANDOM_SEED}" 4096 0
  elif [[ ${MODEL} == "Longformer-large" ]]; then
    eval_qa allenai/longformer-large-4096 "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 1 "${RANDOM_SEED}" 4096 0
  fi
}

#evaluate_all


