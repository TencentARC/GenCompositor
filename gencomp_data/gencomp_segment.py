#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gencomp_segment.py — VideoComp dataset curation, Segmentation stage
(reproduces GenCompositor paper Fig.9 / Sect.B.1 step 3).

Reads _labels/<stem>.json from the Labeling stage (primary_noun), and for each
source video:
  1. Grounded SAM2 segments the "most prominent dynamic object" by its noun
     -> per-frame binary mask (native resolution, preserving the motion trajectory)
  2. Reuses GenCompositor utils.video_utils.create_fg_video:
       - mask video (filtered_mask/): native resolution, keeps the original
         trajectory (not centered)
       - foreground video (fg/): 576x576, object centered in each frame
         (removes global position/trajectory), white background
     create_fg_video embeds paper Data Filtering rule (3): touching the border /
     too small / too fragmented -> returns 0 and the case is dropped
  3. source video (filtered_masked_video/): the source video, aligned/copied

Dependencies (all inside this repo):
  - SAM2+GDINO : gencomp_sam2.py (load_sam2 / sam2_segment / read_video_frames, self-contained)
  - centering  : ../utils/video_utils.py (create_fg_video)

Usage (single machine / debug):
  python gencomp_segment.py --video_dir <source_video_dir> [--out_dir D] [--single <stem>] [--limit N]
Multi-node (launched via pdsh):
  python gencomp_segment.py --video_dir D --node_ips ip1,ip2,... --num_gpus 8
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

# ---- Self-contained Grounded SAM2 helper (this folder) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gencomp_sam2 as smb  # load_sam2, sam2_segment, read_video_frames, resolve_rank_from_ips, probe_frames

# ---- GenCompositor foreground-centering util (repo's utils/) ----
_REPO_ROOT = os.path.dirname(_HERE)                 # .../GenCompositor
sys.path.insert(0, os.path.join(_REPO_ROOT, "utils"))
import video_utils as vu  # create_fg_video (576x576 centering + mask video + paper Filtering rule (3))

OUT_DIR_DEFAULT = os.path.join(_HERE, "output")
FG_SIZE = int(os.environ.get("VE_GC_FG_SIZE", "576"))   # foreground centering canvas size (paper: 576)

# make create_fg_video's output size follow FG_SIZE (vu module-level constants)
vu.output_height = FG_SIZE
vu.output_width = FG_SIZE


def labels_dir(out_dir):  return os.path.join(out_dir, "_labels")
def mask_dir(out_dir):    return os.path.join(out_dir, "filtered_mask")
def fg_dir(out_dir):      return os.path.join(out_dir, "fg")
def source_dir(out_dir):  return os.path.join(out_dir, "filtered_masked_video")


def load_label(out_dir, stem):
    p = os.path.join(labels_dir(out_dir), f"{stem}.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def _dump_frames_png(frames, d):
    """list[PIL RGB] -> d/00000.png ... (stored as BGR so cv2.imread reads them back)"""
    os.makedirs(d, exist_ok=True)
    for i, f in enumerate(frames):
        bgr = cv2.cvtColor(np.asarray(f.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(d, f"{i:05d}.png"), bgr)


def _dump_masks_png(masks, d):
    """list[HxW uint8 0/255] -> d/00000.png ... (3-channel)"""
    os.makedirs(d, exist_ok=True)
    for i, m in enumerate(masks):
        cv2.imwrite(os.path.join(d, f"{i:05d}.png"), np.stack([m, m, m], -1))


def segment_one(sam2, video_path, out_dir, force=False):
    """One video: SAM2 segment -> create_fg_video emits mask/fg -> save source. Returns a status string."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    mask_out = os.path.join(mask_dir(out_dir), f"{stem}.mp4")
    fg_out = os.path.join(fg_dir(out_dir), f"{stem}.mp4")
    src_out = os.path.join(source_dir(out_dir), f"{stem}.mp4")

    if not force and os.path.exists(mask_out) and os.path.exists(fg_out) and os.path.exists(src_out):
        return "skip-done"

    lab = load_label(out_dir, stem)
    if lab is None:
        return "no-label"
    if lab.get("skip") is True:
        return "label-null"      # Filtering rule (1): no significant object
    noun = (lab.get("primary_noun") or "").strip()
    if not noun:
        return "no-noun"

    # read frames + SAM2 segmentation
    try:
        frames, fps = smb.read_video_frames(video_path)
    except Exception as e:
        return f"read-fail:{type(e).__name__}"
    masks, n_boxes = smb.sam2_segment(sam2, frames, noun)
    if n_boxes == 0:
        return "gdino-miss"      # object not detected in the first frame
    white = sum(1 for m in masks if m.max() > 0)
    if white == 0:
        return "empty-mask"

    # write PNGs -> create_fg_video (emits mask video + centered fg video; embeds Filtering rule (3))
    tmp = tempfile.mkdtemp(prefix="gc_seg_")
    try:
        ori_d = os.path.join(tmp, "ori"); msk_d = os.path.join(tmp, "msk")
        _dump_frames_png(frames, ori_d)
        _dump_masks_png(masks, msk_d)
        os.makedirs(mask_dir(out_dir), exist_ok=True)
        os.makedirs(fg_dir(out_dir), exist_ok=True)
        ok = vu.create_fg_video(ori_d, msk_d, mask_out, fg_out, frame_rate=int(round(fps)))
        if ok != 1:
            # dropped by rule (3): remove any partial outputs
            for p in (mask_out, fg_out):
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass
            return "filtered-out"   # border-touching / too small / too fragmented
        # source: write out the source frames as-is (aligned res/frames/fps with the mask)
        os.makedirs(source_dir(out_dir), exist_ok=True)
        _write_source(frames, src_out, fps)
        return "ok"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_source(frames, out_path, fps):
    import imageio.v2 as iio
    w = iio.get_writer(out_path, codec="libx264", pixelformat="yuv420p",
                       quality=8, fps=fps, ffmpeg_params=["-movflags", "+faststart"])
    for f in frames:
        w.append_data(np.asarray(f.convert("RGB")))
    w.close()


def collect_stems(video_dir, out_dir, limit=0, single=None):
    vids = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if single:
        vids = [v for v in vids if os.path.splitext(os.path.basename(v))[0] == single]
    if limit > 0:
        vids = vids[:limit]
    return vids


def run_single_process(video_dir, out_dir, limit=0, single=None, force=False):
    """Single process (single GPU / debug): load SAM2 once, process serially."""
    vids = collect_stems(video_dir, out_dir, limit, single)
    print(f"[Segment] to process: {len(vids)} videos  FG_SIZE={FG_SIZE}", flush=True)
    if not vids:
        return {}
    sam2 = smb.load_sam2()
    print("[Segment] SAM2+GDINO ready", flush=True)
    stats = {}
    for i, v in enumerate(vids):
        st = segment_one(sam2, v, out_dir, force)
        stats[st] = stats.get(st, 0) + 1
        print(f"  [{i+1}/{len(vids)}] {os.path.basename(v)} -> {st}", flush=True)
    print(f"[Segment] done stats: {stats}", flush=True)
    return stats


# ============================================================
# Multi-node / multi-GPU: one process per GPU; after spawn each loads SAM2
# and processes its own case list.
# ============================================================
def gpu_worker(gpu_id, video_list, out_dir, force):
    """One GPU process: bind device -> load SAM2 -> process assigned videos serially."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tag = f"[GPU{gpu_id}]"
    print(f"{tag} loading SAM2+GDINO ...", flush=True)
    sam2 = smb.load_sam2()
    print(f"{tag} ready, {len(video_list)} videos", flush=True)
    stats = {}
    for i, v in enumerate(video_list):
        try:
            st = segment_one(sam2, v, out_dir, force)
        except Exception as e:
            import traceback
            print(f"{tag} ERR {os.path.basename(v)}: {e}", flush=True)
            traceback.print_exc()
            st = f"exc:{type(e).__name__}"
        stats[st] = stats.get(st, 0) + 1
        print(f"{tag} [{i+1}/{len(video_list)}] {os.path.basename(v)} -> {st}", flush=True)
    print(f"{tag} done: {stats}", flush=True)


def run_multi(video_dir, out_dir, num_gpus, node_ips, node_rank, num_nodes, limit=0, force=False):
    """Multi-node / multi-GPU: node-level LPT balance -> skip finished -> GPU-level LPT -> spawn one proc per GPU."""
    import torch.multiprocessing as mp

    # locate this host's rank automatically from --node_ips
    if node_ips:
        rk, nn = smb.resolve_rank_from_ips(node_ips)
        if rk is not None:
            node_rank, num_nodes = rk, nn
            print(f"[rank-remap] this host rank={rk}/{nn}", flush=True)

    vids = collect_stems(video_dir, out_dir, limit)
    print(f"[Segment] total {len(vids)} videos, node {node_rank}/{num_nodes}, {num_gpus} GPU/node", flush=True)

    # node-level LPT (balance by frame count)
    if num_nodes > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=32) as ex:
            fc = list(ex.map(lambda v: smb.probe_frames(v), vids))
        order = sorted(range(len(vids)), key=lambda i: -fc[i])
        buckets = [[] for _ in range(num_nodes)]
        loads = [0] * num_nodes
        for idx in order:
            b = min(range(num_nodes), key=lambda x: loads[x])
            buckets[b].append(vids[idx]); loads[b] += fc[idx]
        vids = buckets[node_rank]
        print(f"this node (rank={node_rank}) handles {len(vids)} videos", flush=True)

    # skip finished (all three videos present)
    if not force:
        def done(v):
            s = os.path.splitext(os.path.basename(v))[0]
            return (os.path.exists(os.path.join(mask_dir(out_dir), f"{s}.mp4"))
                    and os.path.exists(os.path.join(fg_dir(out_dir), f"{s}.mp4"))
                    and os.path.exists(os.path.join(source_dir(out_dir), f"{s}.mp4")))
        vids = [v for v in vids if not done(v)]
    print(f"to process: {len(vids)} videos", flush=True)
    if not vids:
        print("nothing to process, exit", flush=True); return

    # GPU-level LPT
    ng = min(num_gpus, len(vids))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=32) as ex:
        lfc = list(ex.map(lambda v: smb.probe_frames(v), vids))
    order = sorted(range(len(vids)), key=lambda i: -lfc[i])
    gpu_vids = [[] for _ in range(ng)]
    gpu_loads = [0] * ng
    for i in order:
        g = min(range(ng), key=lambda x: gpu_loads[x])
        gpu_vids[g].append(vids[i]); gpu_loads[g] += lfc[i]
    for g in range(ng):
        print(f"  GPU {g}: {len(gpu_vids[g])} videos, {gpu_loads[g]} frames", flush=True)

    mp.set_start_method("spawn", force=True)
    procs = []
    for g in range(ng):
        if gpu_vids[g]:
            p = mp.Process(target=gpu_worker, args=(g, gpu_vids[g], out_dir, force))
            p.start(); procs.append(p)
    for p in procs:
        p.join()
    print(f"\nthis node finished! output -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="VideoComp Segmentation (SAM2 segment + foreground centering + three videos)")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--single", default=None, help="process a single stem (single-process debug)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    # multi-node / multi-GPU
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--node_ips", type=str, default="", help="comma-separated IP list; locate this host's rank by its IP")
    ap.add_argument("--num_nodes", type=int, default=1)
    ap.add_argument("--node_rank", type=int, default=0)
    args = ap.parse_args()

    # --single, or num_gpus<=1 and non-multi-node -> single process; otherwise multi-node/GPU spawn
    if args.single or (args.num_gpus <= 1 and not args.node_ips and args.num_nodes <= 1):
        run_single_process(args.video_dir, args.out_dir, args.limit, args.single, args.force)
    else:
        run_multi(args.video_dir, args.out_dir, args.num_gpus, args.node_ips,
                  args.node_rank, args.num_nodes, args.limit, args.force)


if __name__ == "__main__":
    main()
