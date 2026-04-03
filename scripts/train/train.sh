#!/usr/bin/env bash
# Run RNN pretraining, then residual / PMM training (defaults match the Python CLIs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RNN_PY="${SCRIPT_DIR}/rnn_train.py"
RESIDUAL_PY="${SCRIPT_DIR}/residual_train.py"

# Defaults aligned with argparse defaults in rnn_train.py / residual_train.py
EPISODES_PKL="data/episodes_with_states.pkl"
RNN_LOSS_MODE="all"
RNN_CHECKPOINT_OUT="results/checkpoints/rnn_pretrained.pt"
RESIDUAL_COS_TAU="0.6"
# RESIDUAL_TRAINED_PT=""  # default None: omit --trained-pt to train residual from scratch

echo "Working directory: ${REPO_ROOT}"
echo ""

# ---------------------------------------------------------------------------
# 1) RNN pretrain (rnn_train.py)
#    CLI defaults: --pkl, --loss-mode (other hyperparams are fixed in main(): e.g. epochs=32,
#    steps=128, batch 512, lr=3e-4, state_embed_dim=16, h_dim=64, seed=42)
# ---------------------------------------------------------------------------
echo "========== Step 1/2: RNN pretraining =========="
python "${RNN_PY}" \
  --pkl "${EPISODES_PKL}" \
  --loss-mode "${RNN_LOSS_MODE}"

echo ""

# ---------------------------------------------------------------------------
# 2) Residual network + PMM (residual_train.py)
#    CLI defaults: --input-pkl, --rnn-weights, --cos-tau
#    --trained-pt default is None (no resume); add --trained-pt path to continue.
# ---------------------------------------------------------------------------
echo "========== Step 2/2: Residual + PMM training =========="
python "${RESIDUAL_PY}" \
  --input-pkl "${EPISODES_PKL}" \
  --rnn-weights "${RNN_CHECKPOINT_OUT}" \
  --cos-tau "${RESIDUAL_COS_TAU}"

echo ""
echo "Done. Check results/ and results/checkpoints/."
