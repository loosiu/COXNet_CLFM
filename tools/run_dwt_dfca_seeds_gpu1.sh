#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${DWT_DFCA_PYTHON:-/home/viplab/anaconda3/envs/coxmamba/bin/python}"
config_path="configs/coxnet/dwt_dfca/DWT_DFCA.py"
relative_work_root="work_dir/coxmamba/rgbtdroneperson/dwt_dfca"
work_root="${repo_root}/${relative_work_root}"
data_root="${MMDET_DATASETS:-/data/siwoo/COXNet-release/data/RGBTDronePerson/}"
dry_run="${DWT_DFCA_DRY_RUN:-0}"
seeds=(0 1 2)

export CUDA_VISIBLE_DEVICES="${DWT_DFCA_CUDA_VISIBLE_DEVICES:-1}"
export MMDET_DATASETS="${data_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${dry_run}" != "1" ]]; then
    exec 9>/tmp/coxnet_dwt_dfca_gpu1.lock
    if ! flock -n 9; then
        echo "GPU 1 DWT-DFCA lock is already held" >&2
        exit 1
    fi
fi

cd "${repo_root}"
for seed in "${seeds[@]}"; do
    relative_work_dir="${relative_work_root}/seed${seed}"
    work_dir="${work_root}/seed${seed}"
    command=(
        "${python_bin}" tools/train.py
        --config "${config_path}"
        --work-dir "${work_dir}"
        --gpu-id 0
        --seed "${seed}"
        --deterministic
        --auto-resume
    )

    printf 'RUN seed=%s work_dir=%s command=' "${seed}" "${relative_work_dir}"
    printf '%q ' "${command[@]}"
    printf '\n'

    if [[ "${dry_run}" == "1" ]]; then
        continue
    fi

    mkdir -p "${work_dir}"
    "${command[@]}" 2>&1 | tee "${work_dir}/console.log"
done
