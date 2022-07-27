
export CUDA_VISIBLE_DEVICES=0

python -u eval_ps_ranking.py \
--full_run_mode \
--task phrase_similarity \
--result_dir ../results/phrase_similarity/ranking/ \
#--contextual