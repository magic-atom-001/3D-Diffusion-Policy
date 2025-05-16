#!/bin/bash

# Usage example:
# bash scripts/inference_policy.sh dp3 realdex_drill 0112 0 0

alg_name=${1}               # e.g. dp3
task_name=${2}              # e.g. realdex_drill
addition_info=${3}          # e.g. 0112
seed=${4}                   # e.g. 0
gpu_id=${5}                 # e.g. 0

config_name=${alg_name}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[32m[INFO] 推理开始：${exp_name} (GPU ${gpu_id})\033[0m"

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# python inference_realrobot.py \
#     --config-name=${config_name}.yaml \
#     +workspace=real_robot_workspace \
#     training.device="cuda:0" \
#     training.seed=${seed} \
#     exp_name=${exp_name} \
#     hydra.run.dir=${run_dir} \
#     +checkpoint.ckpt_name=latest.ckpt

python inference_realrobot.py \
    --config-name=dp3.yaml \
    task=realdex_drill \
    hydra.run.dir=${run_dir} \
    +checkpoint.ckpt_name=latest.ckpt
