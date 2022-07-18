
function train_qa() {

  local MODEL_BASE=$1
  local DATASET=$2
  local DATASET_CONFIG=$3
  local OUTPUT_DIR=$4
  local TRAIN_BATCH_SIZE=$5
  local EVAL_BATCH_SIZE=$6
  local SEED=$7
  local MAX_SEQ_LENGTH=$8
  local GPU=$9

  local OUTPUT_DIR=${OUTPUT_DIR}/${DATASET}/${DATASET_CONFIG}/qa/${MODEL_BASE}/finetuned

  mkdir -p $OUTPUT_DIR
  export CUDA_VISIBLE_DEVICES=$GPU

  params=()
  params+=(--do_train)
  params+=(--num_train_epochs 2.0)
#  params+=(--save_steps 100000)

  params+=(--model_name_or_path "${MODEL_BASE}")
  params+=(--do_eval)
  params+=(--dataset_name PiC/"${DATASET}")
  params+=(--dataset_config_name "${DATASET_CONFIG}")
  params+=(--per_device_train_batch_size "${TRAIN_BATCH_SIZE}")
  params+=(--per_device_eval_batch_size "${EVAL_BATCH_SIZE}")
  params+=(--learning_rate 3e-5)
  params+=(--max_seq_length "${MAX_SEQ_LENGTH}")
  params+=(--doc_stride 128)
  params+=(--output_dir "${OUTPUT_DIR}")
  params+=(--overwrite_output_dir)
  params+=(--overwrite_cache)
  params+=(--seed "${SEED}")

  params+=(--evaluation_strategy "steps")
  params+=(--metric_for_best_model "eval_exact_match")
  params+=(--load_best_model_at_end True)

  params+=(--report_to wandb)
  params+=(--run_name "${DATASET}"/"${DATASET_CONFIG}"/qa/"${MODEL_BASE}"/finetuned)

#  Default hyperparameters
#  eval_steps=logging_steps
#  save_strategy="steps"
#  save_steps=500
#  logging_strategy="steps"
#  logging_steps=500

  echo "${params[@]}"
  nohup python -u run_qa.py "${params[@]}" > $OUTPUT_DIR/finetune_logs.txt &
}

function do_finetuning() {
  echo "*** FINETUNING STARTED ***"

  local DATASET="phrase_retrieval"  # ["phrase_retrieval", "phrase_sense_disambiguation"]
  local DATASET_CONFIG="PR-pass"                      # "PR-pass" OR "PR-page" for "phrase_retrieval" else ""
  local OUTPUT_DIR=../results
  local RANDOM_SEED=42

  train_qa bert-base-uncased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 8 "${RANDOM_SEED}" 512 0
  train_qa bert-large-uncased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 8 "${RANDOM_SEED}" 512 1
  train_qa allenai/longformer-base-4096 "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 2 2 "${RANDOM_SEED}" 4096 2
  train_qa allenai/longformer-large-4096 "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 1 1 "${RANDOM_SEED}" 4096 3
  train_qa sentence-transformers/bert-base-nli-stsb-mean-tokens "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 8 "${RANDOM_SEED}" 512 4
  train_qa whaleloops/phrase-bert "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 8 "${RANDOM_SEED}" 512 5
  train_qa SpanBERT/spanbert-base-cased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 8 "${RANDOM_SEED}" 512 6
  train_qa princeton-nlp/sup-simcse-bert-base-uncased "${DATASET}" "${DATASET_CONFIG}" "${OUTPUT_DIR}" 8 8 "${RANDOM_SEED}" 512 7
}

do_finetuning


