#!/bin/bash -l
#SBATCH -J n2s_train_and_denoise_reconst
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

###############################################################################
# USER-EDITABLE SETTINGS (KEEP THESE AT THE TOP)
###############################################################################

# ---- REQUIRED ABSOLUTE PATHS (cluster) ----
# Absolute path to the repo on the cluster filesystem (must contain train_noise2self.py)
REPO_DIR="/raven/u/kumate/Hackathon/noise2self"

# Absolute path to input dataset (.h5)
H5_PATH="/ptmp/kumate/Hackathon_data/dataset_4/processed_win00_left512.h5"

# Absolute run directory (checkpoints + optional output). DO NOT include SLURM job id here.
# Change this to a meaningful folder name before submitting.
RUN_DIR="/ptmp/kumate/Hackathon_data/hackathon_runs/final_run_2"

# Where Slurm stdout/stderr files should be written (absolute).
# Recommended: keep logs inside the run directory so everything stays together.
LOG_DIR="${RUN_DIR}/logs"

# ---- Training hyperparams ----
MODEL="unet"        # unet | baby-unet | convolution | dncnn
PATCH="128"         # 0 for full-frame, else 128 recommended
BATCH="16"          # reduce if OOM
EPOCHS="10"
LR="1e-4"
WEIGHT_DECAY="0.0"
WORKERS="4"
SEED="0"

# ---- Masking ----
MASKER_WIDTH="4"
MASKER_MODE="interpolate"  # interpolate | zero
INCLUDE_MASK="0"           # 1 -> add --include-mask

# ---- Validation + checkpoints ----
VAL_FRAC="0.05"
VAL_MAX_BATCHES="50"             # 0 -> full val pass
EARLY_STOP_PATIENCE="100"        # 0 -> disable early stop
EARLY_STOP_MIN_DELTA="1e-6"
SAVE_BEST="1"                    # 1 -> write ckpt_best.pt (requires VAL_FRAC>0)

# ---- Reconstructable ("counts-space") training ----
# For denoised output to be meaningful counts: normalize must be "none"
# and integer inputs must NOT be scaled to [0,1].
NORMALIZE="none"         # none|minmax|percentile
SCALE_INTEGERS="0"       # 0 -> --no-scale-integers ; 1 -> --scale-integers

# ---- Denoise output (.h5) ----
# If empty, defaults to: ${RUN_DIR}/denoised.h5
OUT_H5=""

# ---- Checkpoint selection for denoising ----
# If empty, the script will prefer ${RUN_DIR}/ckpt_best.pt, else the newest ckpt_epoch*.pt
CKPT_PATH=""

# ---- Runtime toggles ----
DEVICE="cuda"
USE_AMP="1"              # 1 -> add --amp
STDOUT_EPOCH_ONLY="1"    # 1 -> add --stdout-epoch-only (smaller .out)
USE_TQDM="1"             # 1 -> add --tqdm (progress to .err)

###############################################################################
# END USER-EDITABLE SETTINGS
###############################################################################

die() { echo "ERROR: $*" 1>&2; exit 2; }

require_abs() {
  local name="$1"
  local val="$2"
  if [[ -z "${val}" ]]; then
    die "${name} is empty (must be an absolute path)"
  fi
  if [[ "${val}" != /* ]]; then
    die "${name} must be an absolute path starting with '/': got '${val}'"
  fi
}

if [[ "${RUN_DIR}" == *"CHANGE_ME"* ]]; then
  die "RUN_DIR still contains 'CHANGE_ME'. Edit RUN_DIR at the top of this script to an absolute path you choose."
fi

require_abs "REPO_DIR" "${REPO_DIR}"
require_abs "H5_PATH" "${H5_PATH}"
require_abs "RUN_DIR" "${RUN_DIR}"
require_abs "LOG_DIR" "${LOG_DIR}"

SAVE_DIR="${RUN_DIR}"

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

# Slurm logs: keep them in LOG_DIR regardless of where you submit from.
OUT_FILE="${LOG_DIR}/slurm-${SLURM_JOB_ID:-nojob}.out"
ERR_FILE="${LOG_DIR}/slurm-${SLURM_JOB_ID:-nojob}.err"
exec >"${OUT_FILE}" 2>"${ERR_FILE}"

echo "pwd(before): $(pwd)"
cd "${REPO_DIR}"
echo "pwd(repo): $(pwd)"

# Optional: cluster modules / env activation (edit if your cluster differs).
if command -v module &> /dev/null; then
  module purge
  module load gcc/13 cuda/12.6
fi
if [[ -f "/u/kumate/.venvs/surrogate-cu/bin/activate" ]]; then
  source /u/kumate/.venvs/surrogate-cu/bin/activate
fi

echo "python: $(python -V)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# Print parameter counts early (so it appears near the top of the .out file)
python - <<'PY'
import os
import torch

model_name = os.environ.get("MODEL", "unet").lower()
include_mask = os.environ.get("INCLUDE_MASK", "0") == "1"

# Train script defaults: in_ch=1, out_ch=1; include_mask adds an extra input channel.
in_ch = 1 + (1 if include_mask else 0)
out_ch = 1

if model_name == "unet":
    from models.unet import Unet
    model = Unet(in_ch, out_ch)
elif model_name in {"babyunet", "baby-unet"}:
    from models.babyunet import BabyUnet
    model = BabyUnet(in_ch, out_ch)
elif model_name in {"conv", "convolution", "singleconv"}:
    from models.singleconv import SingleConvolution
    model = SingleConvolution(in_ch, out_ch, width=3)
elif model_name == "dncnn":
    from models.dncnn import DnCNN
    model = DnCNN(in_ch, out_ch)
else:
    raise SystemExit(f"Unknown MODEL={model_name!r} for param count.")

total = sum(p.numel() for p in model.parameters())
learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("==== PARAM COUNT (pre-train) ====")
print(f"model_name: {model_name}")
print(f"in_channels: {in_ch} out_channels: {out_ch} include_mask: {include_mask}")
print(f"total_params: {total}")
print(f"learnable_params: {learnable}")
print("================================")
PY

if [[ -z "${OUT_H5}" ]]; then
  OUT_H5="${RUN_DIR}/denoised.h5"
fi
require_abs "OUT_H5" "${OUT_H5}"

INCLUDE_MASK_FLAG=""
if [[ "${INCLUDE_MASK}" == "1" ]]; then
  INCLUDE_MASK_FLAG="--include-mask"
fi

SCALE_INTEGERS_FLAG="--scale-integers"
if [[ "${SCALE_INTEGERS}" == "0" ]]; then
  SCALE_INTEGERS_FLAG="--no-scale-integers"
fi

AMP_FLAG=""
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_FLAG="--amp"
fi

STDOUT_EPOCH_ONLY_FLAG=""
if [[ "${STDOUT_EPOCH_ONLY}" == "1" ]]; then
  STDOUT_EPOCH_ONLY_FLAG="--stdout-epoch-only"
fi

TQDM_FLAG=""
if [[ "${USE_TQDM}" == "1" ]]; then
  TQDM_FLAG="--tqdm"
fi

SAVE_BEST_FLAG=""
if [[ "${SAVE_BEST}" == "1" ]]; then
  SAVE_BEST_FLAG="--save-best"
fi

echo "==== SETTINGS ===="
echo "REPO_DIR: ${REPO_DIR}"
echo "H5_PATH: ${H5_PATH}"
echo "RUN_DIR: ${RUN_DIR}"
echo "SAVE_DIR: ${SAVE_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "OUT_H5: ${OUT_H5}"
echo "train: model=${MODEL} patch=${PATCH} batch=${BATCH} epochs=${EPOCHS} lr=${LR} wd=${WEIGHT_DECAY} workers=${WORKERS} seed=${SEED}"
echo "masker: width=${MASKER_WIDTH} mode=${MASKER_MODE} include_mask=${INCLUDE_MASK}"
echo "val: frac=${VAL_FRAC} max_batches=${VAL_MAX_BATCHES} early_stop_patience=${EARLY_STOP_PATIENCE} min_delta=${EARLY_STOP_MIN_DELTA} save_best=${SAVE_BEST}"
echo "reconst: normalize=${NORMALIZE} scale_integers=${SCALE_INTEGERS}"
echo "infer: device=${DEVICE} amp=${USE_AMP}"
echo "=================="

echo "==== TRAIN ===="
python -u "${REPO_DIR}/train_noise2self.py" \
  --h5 "${H5_PATH}" \
  --device "${DEVICE}" \
  --model "${MODEL}" \
  --patch "${PATCH}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --normalize "${NORMALIZE}" \
  ${SCALE_INTEGERS_FLAG} \
  --masker-width "${MASKER_WIDTH}" \
  --masker-mode "${MASKER_MODE}" \
  ${INCLUDE_MASK_FLAG} \
  --workers "${WORKERS}" \
  --seed "${SEED}" \
  ${AMP_FLAG} \
  ${STDOUT_EPOCH_ONLY_FLAG} \
  ${TQDM_FLAG} \
  --val-frac "${VAL_FRAC}" \
  --val-max-batches "${VAL_MAX_BATCHES}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
  ${SAVE_BEST_FLAG} \
  --save-dir "${SAVE_DIR}"

echo "==== SELECT CHECKPOINT FOR DENOISE ===="
if [[ -z "${CKPT_PATH}" ]]; then
  if [[ -f "${SAVE_DIR}/ckpt_best.pt" ]]; then
    CKPT_PATH="${SAVE_DIR}/ckpt_best.pt"
  else
    # newest epoch checkpoint
    CKPT_PATH="$(ls -1t "${SAVE_DIR}"/ckpt_epoch*.pt 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ -z "${CKPT_PATH}" ]]; then
  die "Could not find a checkpoint to denoise with. Expected ckpt_best.pt or ckpt_epoch*.pt in ${SAVE_DIR}."
fi
require_abs "CKPT_PATH" "${CKPT_PATH}"
if [[ ! -f "${CKPT_PATH}" ]]; then
  die "Checkpoint not found: ${CKPT_PATH}"
fi
echo "CKPT_PATH: ${CKPT_PATH}"

echo "==== DENOISE -> HDF5 ===="
python -u "${REPO_DIR}/scripts/denoise_h5_stack.py" \
  --h5 "${H5_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --out "${OUT_H5}" \
  --device "${DEVICE}" \
  --batch "32" \
  ${AMP_FLAG} \
  --require-reconstructable

echo "DONE"


