#!/usr/bin/env bash
# One-shot conda env for ENAP_public: Python 3.10, ManiSkill, deps, custom PegInsertionSide.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPLACE_SRC="${REPO_ROOT}/peg_insertion_side_replace.py"
ENV_NAME="enap_env"

if [[ ! -f "${REPLACE_SRC}" ]]; then
  echo "ERROR: Missing ${REPLACE_SRC} (run this script from the repo root)."
  exit 1
fi

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install Miniconda or Anaconda first."
  exit 1
fi

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "Conda env '${ENV_NAME}' already exists; skipping create."
  echo "To recreate: conda env remove -n ${ENV_NAME} -y && re-run this script."
else
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

echo "Installing mani_skill, torch, hdbscan, scikit-learn..."
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install --upgrade mani_skill torch
conda run -n "${ENV_NAME}" pip install hdbscan scikit-learn

TARGET="$(conda run -n "${ENV_NAME}" python -c \
  "import mani_skill.envs.tasks.tabletop.peg_insertion_side as m; print(m.__file__)")"
cp "${REPLACE_SRC}" "${TARGET}"
echo ""
echo "Patched custom env file:"
echo "  ${TARGET}"
echo ""
echo "Activate: conda activate ${ENV_NAME}"
