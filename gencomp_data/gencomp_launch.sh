#!/usr/bin/env bash
# ============================================================
# gencomp_launch.sh — Multi-node / multi-GPU launcher for VideoComp dataset
#                     curation (reproducing the GenCompositor Fig.9 pipeline).
#
# Two stages:
#   Stage1 Labeling     : rank0 only runs gencomp_label.py (Qwen VL watches the
#                         video -> most prominent dynamic-object nouns; network-only)
#   Stage2 Segmentation : all nodes x GPUs run gencomp_segment.py in parallel
#                         (Grounded SAM2 -> foreground centering -> three videos;
#                          LPT-balanced by frame count + skip already-finished)
#
# Required env (set by scheduler):
#   NODE_IP_LIST   — "ip1:8,ip2:8,..." (with the :8 suffix)
#   HOST_NUM       — total node count (auto-scales, never hardcoded)
# Optional env:
#   GPUS_PER_NODE      — default 8
#   VE_GC_VIDEO_DIR    — source video dir (required; or pass --video_dir)
#   VE_GC_OUT_DIR      — output dir (default: gencomp_data/output)
#   VE_GC_FG_SIZE      — foreground centering canvas size (default 576)
#   VE_GC_LIMIT        — >0 to process only the first N videos (smoke test)
#   VE_GC_LABEL_WORKERS— labeling concurrency (default 16)
#   GC_SKIP_LABEL      — =1 to skip Stage1 (when labels already exist)
#
# Usage (head node):
#   export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
#   VE_GC_VIDEO_DIR=/path/to/source_videos bash gencomp_launch.sh
# Smoke test (16 videos):
#   VE_GC_VIDEO_DIR=... VE_GC_LIMIT=16 bash gencomp_launch.sh
# ============================================================

set -e

if [ -z "${NODE_IP_LIST}" ] || [ -z "${HOST_NUM}" ]; then
    echo "[!] NODE_IP_LIST / HOST_NUM not set"; exit 1
fi
if [ -z "${VE_GC_VIDEO_DIR}" ]; then
    echo "[!] VE_GC_VIDEO_DIR not set (source video dir)"; exit 1
fi

ALL_IPS_CSV=$(echo "${NODE_IP_LIST}" | sed 's/:8//g')
IFS=',' read -r -a ALL_IPS <<< "${ALL_IPS_CSV}"
TOTAL_NODES=${#ALL_IPS[@]}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

PYTHON_BIN="python"
GC_DIR="./GenCompositor-main/gencomp_data"
# Repo root (contains utils/video_utils.py and gencomp_data/). Default = parent of GC_DIR.
WORK_ROOT="${WORK_ROOT:-$(dirname "${GC_DIR}")}"

OUT_DIR="${VE_GC_OUT_DIR:-${GC_DIR}/output}"
FG_SIZE="${VE_GC_FG_SIZE:-576}"
LABEL_WORKERS="${VE_GC_LABEL_WORKERS:-16}"
LIMIT_FLAG=""
[ -n "${VE_GC_LIMIT}" ] && [ "${VE_GC_LIMIT}" -gt 0 ] 2>/dev/null && LIMIT_FLAG="--limit ${VE_GC_LIMIT}"

echo "[$(date)] nodes=${TOTAL_NODES} x ${GPUS_PER_NODE} GPU"
echo "[$(date)] source: ${VE_GC_VIDEO_DIR}"
echo "[$(date)] output: ${OUT_DIR}   FG_SIZE=${FG_SIZE}"

# ---- Stage 1: Labeling (head/rank0 only, network-only) ----
if [ "${GC_SKIP_LABEL:-0}" != "1" ]; then
    echo "[$(date)] === Stage1 Labeling (Qwen VL, rank0 only) ==="
    cd "${WORK_ROOT}"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    VE_GC_FG_SIZE=${FG_SIZE} PYTHONUNBUFFERED=1 ${PYTHON_BIN} -u "${GC_DIR}/gencomp_label.py" \
        --video_dir "${VE_GC_VIDEO_DIR}" --out_dir "${OUT_DIR}" \
        --workers ${LABEL_WORKERS} ${LIMIT_FLAG}
    echo "[$(date)] Stage1 done."
else
    echo "[$(date)] skip Stage1 (GC_SKIP_LABEL=1)"
fi

# ---- Stage 2: Segmentation (all nodes x GPUs, pdsh fan-out) ----
echo "[$(date)] === Stage2 Segmentation (Grounded SAM2, all nodes x ${GPUS_PER_NODE} GPU) ==="
pdsh -R ssh -f 256 -w "${ALL_IPS_CSV}" \
    "cd ${WORK_ROOT} && \
     unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
     VE_GC_FG_SIZE=${FG_SIZE} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
     ${PYTHON_BIN} -u ${GC_DIR}/gencomp_segment.py \
        --video_dir ${VE_GC_VIDEO_DIR} \
        --out_dir ${OUT_DIR} \
        --num_nodes ${TOTAL_NODES} \
        --node_rank 0 \
        --node_ips ${ALL_IPS_CSV} \
        --num_gpus ${GPUS_PER_NODE} \
        ${LIMIT_FLAG}" 2>&1

echo "[$(date)] all gencomp nodes finished. output -> ${OUT_DIR}/{filtered_masked_video,filtered_mask,fg}"
