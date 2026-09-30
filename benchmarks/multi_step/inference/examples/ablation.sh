CUDA_VISIBLE_DEVICES=0 bash benchmarks/multi_step/inference/run_ablation.sh \
--language en \
--ablation-mode all \
--batch-size 2 \
--model-path "${ANESTRACE_MODEL_ROOT:-./checkpoints}/MiliLab/Morpheus-32B"
