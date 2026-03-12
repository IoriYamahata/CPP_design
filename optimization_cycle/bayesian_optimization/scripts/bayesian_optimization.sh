#!/usr/bin/env bash

set -euo pipefail

project_dir="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

source "${project_dir}/directory_setting.sh"

python "${BAYESIAN_OPTIMIZATION_DIR}/scripts/make_optimize_input.py"
python "${BAYESIAN_OPTIMIZATION_DIR}/scripts/optimization.py"
