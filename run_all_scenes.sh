#!/usr/bin/env bash
# Run train.py → render.py → metrics.py for selected 4DGS test-dataset scenes.
#
# Usage:
#   bash run_all_scenes.sh                                        # all scenes, baseline
#   bash run_all_scenes.sh --scene 1                             # scene 1 only (by number)
#   bash run_all_scenes.sh --scene scene3_collision              # scene by full name
#   bash run_all_scenes.sh --scene 1 --scene 3                  # multiple scenes
#
# Training modes (can be combined; each appends a suffix to the output dir):
#   --fg_mask_loss                  foreground-weighted L1 loss (uses per-frame masks)
#   --fg_bg_weight 0.05             background pixel weight for fg_mask_loss (default 0.05)
#   --drift_loss                    trajectory-guided Gaussian anchor loss (shape-aware SDF)
#   --lambda_drift 0.05             weight for drift loss (default 0.05)
#   --scale_reg                     scale regularisation to suppress needle Gaussians
#   --lambda_scale_reg 0.01         weight for scale regularisation (default 0.01)
#
# Logs: each stage writes to output/<expname>/logs/{train,render,metrics}.log
#
# Examples:
#   bash run_all_scenes.sh --fg_mask_loss                        # fg-mask mode, all scenes
#   bash run_all_scenes.sh --fg_mask_loss --fg_bg_weight 0.01   # lower bg weight
#   bash run_all_scenes.sh --drift_loss --scale_reg              # drift+scale mode
#   bash run_all_scenes.sh --fg_mask_loss --drift_loss --scale_reg  # combined (best predicted)
#   bash run_all_scenes.sh --scene 2 --drift_loss               # scene 2, drift only
#
# Mode flags are forwarded to train.py and appended to the expname so all
# runs coexist under separate output directories.

set -euo pipefail

DATASET_ROOT="/root/4dgs-test-dataset/dnerf_dataset"
ARGS_SRC="/root/4dgs-test-dataset/arguments"
ARGS_DST="arguments/dnerf"
PORT=6017

# Prevent libgomp from spawning one thread per CPU, which exhausts thread
# resources and causes heap corruption / segfaults in render.py.
export OMP_NUM_THREADS=4

ALL_SCENES=(
    "scene1_close_proximity"
    "scene2_identical_objects"
    "scene3_collision"
    "scene4_occlusion"
    "scene5_rapid_motion"
    "scene6_scale_change"
    "scene7_deformation"
    "scene8_thin_structure"
    "scene9_topology"
    "scene10_texture"
)

# ── Argument parsing ────────────────────────────────────────────────────────
SELECTED_SCENES=()
TRAIN_FLAGS=""
EXPNAME_SUFFIX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)
            SCENE_ARG="$2"
            shift 2
            if [[ "$SCENE_ARG" =~ ^[0-9]+$ ]]; then
                # Numeric: find the scene whose name starts with "scene${N}_"
                for S in "${ALL_SCENES[@]}"; do
                    if [[ "$S" == "scene${SCENE_ARG}_"* ]]; then
                        SELECTED_SCENES+=("$S")
                        break
                    fi
                done
            else
                SELECTED_SCENES+=("$SCENE_ARG")
            fi
            ;;
        --all)
            SELECTED_SCENES=("${ALL_SCENES[@]}")
            shift
            ;;
        --fg_mask_loss)
            TRAIN_FLAGS="$TRAIN_FLAGS --fg_mask_loss"
            EXPNAME_SUFFIX="${EXPNAME_SUFFIX}_fg_mask"
            shift
            ;;
        --drift_loss)
            TRAIN_FLAGS="$TRAIN_FLAGS --drift_loss"
            EXPNAME_SUFFIX="${EXPNAME_SUFFIX}_drift"
            shift
            ;;
        --scale_reg)
            TRAIN_FLAGS="$TRAIN_FLAGS --scale_reg"
            EXPNAME_SUFFIX="${EXPNAME_SUFFIX}_scale_reg"
            shift
            ;;
        --lambda_scale_reg)
            TRAIN_FLAGS="$TRAIN_FLAGS --lambda_scale_reg $2"
            shift 2
            ;;
        --lambda_drift)
            TRAIN_FLAGS="$TRAIN_FLAGS --lambda_drift $2"
            shift 2
            ;;
        --fg_bg_weight)
            TRAIN_FLAGS="$TRAIN_FLAGS --fg_bg_weight $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Valid flags: --scene N|name  --all  --fg_mask_loss  --fg_bg_weight F  --drift_loss  --lambda_drift F  --scale_reg  --lambda_scale_reg F"
            exit 1
            ;;
    esac
done

# Default: run all scenes
if [[ ${#SELECTED_SCENES[@]} -eq 0 ]]; then
    SELECTED_SCENES=("${ALL_SCENES[@]}")
fi

mkdir -p "$ARGS_DST"

# ── Per-scene loop ───────────────────────────────────────────────────────────
for SCENE in "${SELECTED_SCENES[@]}"; do
    EXPNAME="${SCENE}${EXPNAME_SUFFIX}"

    echo ""
    echo "============================================================"
    echo "  Scene : $SCENE"
    echo "  Mode  : ${EXPNAME_SUFFIX:-baseline}"
    echo "  Expname: $EXPNAME"
    echo "============================================================"

    SCENE_DATA="$DATASET_ROOT/$SCENE"
    SCENE_CFG_SRC="$ARGS_SRC/${SCENE}.py"
    SCENE_CFG_DST="$ARGS_DST/${SCENE}.py"
    OUTPUT_DIR="output/$EXPNAME"
    LOG_DIR="$OUTPUT_DIR/logs"

    # Create output and log dirs before training writes there
    mkdir -p "$LOG_DIR"

    # Copy the per-scene config from the dataset arguments folder
    cp "$SCENE_CFG_SRC" "$SCENE_CFG_DST"

    # ── Train ──────────────────────────────────────────────────────
    echo "[train] $EXPNAME  →  $LOG_DIR/train.log"
    python train.py \
        -s "$SCENE_DATA" \
        --port "$PORT" \
        --expname "$EXPNAME" \
        --configs "$SCENE_CFG_DST" \
        $TRAIN_FLAGS \
        2>&1 | tee "$LOG_DIR/train.log"

    # ── Render ─────────────────────────────────────────────────────
    echo "[render] $EXPNAME  →  $LOG_DIR/render.log"
    python render.py \
        --model_path "$OUTPUT_DIR" \
        --configs "$SCENE_CFG_DST" \
        2>&1 | tee "$LOG_DIR/render.log"

    # ── Metrics ────────────────────────────────────────────────────
    echo "[metrics] $EXPNAME  →  $LOG_DIR/metrics.log"
    python metrics.py \
        -m "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_DIR/metrics.log"

    echo "[done] $EXPNAME"
done

echo ""
echo "All scenes complete."
