#!/usr/bin/env bash
# Convenience launcher for a RoboDojo runner that consumes Hy-VLA policy hooks.
#
# Usage:
#   bash robodojo_eval/eval.sh <runner_script> <task_name> <seed> <gpu_id> [extra args...]
#
# The runner script is expected to accept a Hy-VLA config path and policy module
# via common argparse-style flags:
#   --config robodojo_eval/deploy_policy.yml
#   --policy_name robodojo_eval.deploy_policy
#   --task_name <task_name>
#   --seed <seed>
#
# If your RoboDojo checkout uses different flag names, invoke its runner
# directly and point it at robodojo_eval/deploy_policy.yml.

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash robodojo_eval/eval.sh <runner_script> <task_name> <seed> <gpu_id> [extra args...]" >&2
  exit 1
fi

runner_script=${1}
task_name=${2}
seed=${3}
gpu_id=${4}
shift 4

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python "${runner_script}" \
    --config robodojo_eval/deploy_policy.yml \
    --policy_name robodojo_eval.deploy_policy \
    --task_name "${task_name}" \
    --seed "${seed}" \
    "$@"
