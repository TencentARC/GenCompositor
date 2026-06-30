#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gencomp_sam2.py — Self-contained Grounded SAM2 helper for VideoComp dataset curation.

Wraps GroundingDINO (open-vocabulary first-frame detection) + SAM2 (video mask
propagation) into a few simple functions used by gencomp_segment.py. No dependency
on any folder outside this repository — only on a local Grounded-SAM-2 installation,
whose paths are configured via environment variables (see README).

Env (override the default Grounded-SAM-2 layout if needed):
  GSAM2_ROOT     root of the Grounded-SAM-2 repo (contains grounding_dino/, checkpoints/)
  SAM2_PKG       path to the `sam2` python package (if not importable on PYTHONPATH)
  SAM2_CKPT      SAM2 checkpoint (default: $GSAM2_ROOT/checkpoints/sam2.1_hiera_large.pt)
  SAM2_CFG       SAM2 config name (default: configs/sam2.1/sam2.1_hiera_l.yaml)
  GDINO_CFG      GroundingDINO config .py
  GDINO_CKPT     GroundingDINO checkpoint .pth
  GC_BOX_THRESH / GC_TEXT_THRESH   detection thresholds (default 0.3 / 0.25)
"""
from __future__ import annotations
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

# ============================================================
# Grounded-SAM-2 paths (override via env for your own install)
# By default, assume Grounded-SAM-2 sits next to the GenCompositor repo (../Grounded-SAM-2).
# ============================================================
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../GenCompositor
_DEFAULT_GSAM2 = os.path.join(os.path.dirname(_REPO_ROOT), "Grounded-SAM-2")
GSAM2_ROOT = os.environ.get("GSAM2_ROOT", _DEFAULT_GSAM2)
SAM2_PKG = os.environ.get("SAM2_PKG", "")   # empty => assume `sam2` is pip-installed (on PYTHONPATH)
SAM2_CKPT = os.environ.get("SAM2_CKPT", os.path.join(GSAM2_ROOT, "checkpoints", "sam2.1_hiera_large.pt"))
SAM2_CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
GDINO_CFG = os.environ.get(
    "GDINO_CFG", os.path.join(GSAM2_ROOT, "grounding_dino", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"))
GDINO_CKPT = os.environ.get(
    "GDINO_CKPT", os.path.join(GSAM2_ROOT, "gdino_checkpoints", "groundingdino_swint_ogc.pth"))
BOX_THRESH = float(os.environ.get("GC_BOX_THRESH", "0.3"))
TEXT_THRESH = float(os.environ.get("GC_TEXT_THRESH", "0.25"))


# ============================================================
# Read all frames of the source video (native resolution, no crop/resize)
# ============================================================
def read_video_frames(video_path):
    """Read all frames -> (list[PIL RGB] at native resolution, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"video has 0 frames: {video_path}")
    return frames, float(fps)


def probe_frames(video_path):
    """Frame count of a video (used for multi-node / multi-GPU LPT load balancing)."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(1, n)
    except Exception:
        return 100


# ============================================================
# Multi-node: locate this host's rank within the node_ips list by its local IP
# ============================================================
def _local_ips():
    import socket
    import subprocess
    ips = set()
    try:
        out = subprocess.check_output(["hostname", "-I"], timeout=10).decode()
        ips.update(out.split())
    except Exception:
        pass
    try:
        ips.add(socket.gethostbyname(socket.gethostname()))
    except Exception:
        pass
    return {ip.strip() for ip in ips if ip.strip()}


def resolve_rank_from_ips(node_ips_csv):
    ips = [x.strip() for x in node_ips_csv.split(",") if x.strip()]
    if not ips:
        return None, None
    mine = _local_ips()
    for i, ip in enumerate(ips):
        if ip in mine:
            return i, len(ips)
    return None, None


# ============================================================
# Load SAM2 + GroundingDINO (once per process)
# ============================================================
def load_sam2():
    """Load GroundingDINO + SAM2 (image + video predictor). Returns a dict."""
    import torch
    if SAM2_PKG:
        sys.path.insert(0, SAM2_PKG)
    sys.path.insert(0, GSAM2_ROOT)
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    gdino = load_model(GDINO_CFG, GDINO_CKPT, device="cuda")
    vp = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT)
    ip = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT))
    return {"gdino": gdino, "vp": vp, "ip": ip}


# ============================================================
# SAM2 segmentation: GDINO first-frame box + SAM2 video propagation -> per-frame binary mask (native res)
# ============================================================
def sam2_segment(sam2, frames, obj_noun):
    """Run GDINO (first-frame box) + SAM2 video propagation over list[PIL RGB] (native resolution).
    Returns (masks: list[np.uint8 HxW 0/255], n_boxes). masks = raw SAM2 binary segmentation (no dilation)."""
    import torch
    from grounding_dino.groundingdino.util.inference import load_image, predict
    from torchvision.ops import box_convert

    gdino, vp, ip = sam2["gdino"], sam2["vp"], sam2["ip"]
    n = len(frames)
    W0, H0 = frames[0].size   # PIL: (w, h)
    tmp = tempfile.mkdtemp(prefix="gc_sam2_")
    try:
        for i, f in enumerate(frames):
            f.convert("RGB").save(f"{tmp}/{i:05d}.jpg", quality=95)

        caption = obj_noun.strip().rstrip(".") + " ."
        src, t = load_image(f"{tmp}/00000.jpg")
        boxes, _, _ = predict(model=gdino, image=t, caption=caption,
                              box_threshold=BOX_THRESH, text_threshold=TEXT_THRESH)
        masks = [np.zeros((H0, W0), np.uint8) for _ in range(n)]
        n_boxes = len(boxes)
        if n_boxes > 0:
            h, w, _ = src.shape
            bxyxy = box_convert(boxes * torch.Tensor([w, h, w, h]).to(boxes.device),
                                in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            ip.set_image(src)
            im, _, _ = ip.predict(box=bxyxy, multimask_output=False,
                                  point_coords=None, point_labels=None)
            if im.ndim == 4:
                im = im.squeeze(1)
            state = vp.init_state(video_path=tmp)
            for oid, m in enumerate(im, start=1):
                vp.add_new_mask(inference_state=state, frame_idx=0, obj_id=oid, mask=m)
            seg = {}
            for fi, ids, logits in vp.propagate_in_video(state):
                seg[fi] = [(logits[i] > 0).cpu().numpy() for i in range(len(ids))]
            for i in range(n):
                if i in seg:
                    comb = np.zeros((H0, W0), np.uint8)
                    for om in seg[i]:
                        mm = om.squeeze().astype(np.uint8) * 255
                        if mm.shape != (H0, W0):
                            mm = cv2.resize(mm, (W0, H0), interpolation=cv2.INTER_NEAREST)
                        comb = np.maximum(comb, mm)
                    masks[i] = comb
            vp.reset_state(state)
        return masks, n_boxes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
