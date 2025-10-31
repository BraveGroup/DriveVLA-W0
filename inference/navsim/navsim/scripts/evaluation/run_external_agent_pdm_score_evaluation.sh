TRAIN_TEST_SPLIT=navmini
EXPERIMENT_PATH="${1:-/home/hongxiao.yu/LlamaVideo_Official/viz_new/planning_result_drivinggpt_pi0_4prefill_60kiters/results_107-GPT-VA-Tower_checkpoints_0060000.pt}"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=external_agent \
agent.experiment_path=$EXPERIMENT_PATH \
experiment_name=external_agent \
