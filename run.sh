GPU=7
# CUDA_VISIBLE_DEVICES=$GPU bash 12_optimize_rotation_gptq.sh Llama-3-8b 4 4 4
# CUDA_VISIBLE_DEVICES=$GPU bash 20_eval_ptq.sh Llama-3-8b 4 4 4 ./rotation_matrices/Llama-3-8b/gptq_aware/R_base.bin

CUDA_VISIBLE_DEVICES=$GPU bash 10_optimize_rotation.sh Llama-3-8b 4 4 4
CUDA_VISIBLE_DEVICES=$GPU bash 20_eval_ptq.sh Llama-3-8b 4 4 4 ./rotation_matrices/Llama-3-8b/rtn_aware/R_base.bin

# for target_module in q_proj k_proj v_proj o_proj up_proj gate_proj down_proj
# do
#     CUDA_VISIBLE_DEVICES=$GPU bash 13_optimize_rotation_partial_gptq.sh Llama-3-8b 4 4 4 $target_module
#     CUDA_VISIBLE_DEVICES=$GPU bash 21_eval_ptq_partial_gptq.sh Llama-3-8b 4 4 4 ./rotation_matrices/Llama-3-8b/gptq_aware/R_${target_module}.bin $target_module
# done