#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gencomp_label.py — VideoComp dataset curation, Labeling stage
(reproduces GenCompositor paper Fig.9 / Sect.B.1 step 2).

Paper flow: CogVLM describes the video -> QWen identifies the "most prominent
dynamic object" from that description, returning comma-separated nouns, or NULL
if none. We reproduce both steps with a single Qwen vision-language model
(Qwen3.6, which can read video directly):
  Q1 (describe): describe the video in detail, focusing on motion/changes
  Q2 (identify): from the description, name the most prominent dynamic object,
                 returning comma-separated English nouns, or NULL

Output sidecar: OUT_DIR/_labels/<stem>.json
  {stem, description, nouns:[...], primary_noun, skip(bool), reason}
  empty nouns / NULL -> skip=true (paper Data Filtering rule (1): drop videos
  with no significant object)

The sampled clip is fed to Qwen3.6 through an OpenAI-compatible /chat/completions
API using video_url (file://).

Usage:
  python gencomp_label.py --video_dir <source_video_dir> [--out_dir D] [--limit N] [--workers 8] [--force]
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
import re

import requests

# ---- Qwen3.6 vision-language endpoint (OpenAI-compatible /chat/completions) ----
# After serving Qwen3.6 with vLLM, point to it via env vars (see README).
_QWEN_BASE = (os.environ.get("VOE_QWEN_VL_URL", "").strip()
              or os.environ.get("QWEN_ENDPOINT_URL", "").strip()
              or "http://127.0.0.1:9000/v1")
if _QWEN_BASE.endswith("/chat/completions"):
    QWEN_VL_URL = _QWEN_BASE
else:
    QWEN_VL_URL = _QWEN_BASE.rstrip("/") + "/chat/completions"
QWEN_VL_MODEL = os.environ.get("VOE_QWEN_VL_MODEL", "Qwen3.6-27B")

LABEL_FPS = int(os.environ.get("VE_GC_LABEL_FPS", "4"))   # downsample fps fed to Qwen (sparse is enough for description)
SAMPLE_QUALITY = int(os.environ.get("VE_GC_SAMPLE_QUALITY", "8"))  # resampled mp4 quality (imageio 0-10)
TIMEOUT = int(os.environ.get("VE_GC_QWEN_TIMEOUT", "180"))
MAX_RETRIES = int(os.environ.get("VE_GC_QWEN_RETRIES", "3"))
LABEL_VERSION = "gencomp-label-v1"

# Default output dir = ./output next to this script (self-contained; override with --out_dir)
_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def make_fps_sampled(src_mp4, out_mp4, target_fps=LABEL_FPS):
    """Uniformly resample the video to target_fps (picking from source frames) and
    write a small mp4 to feed Qwen. Returns out_mp4 or None."""
    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 2000:
        return out_mp4
    if not (src_mp4 and os.path.exists(src_mp4)):
        return None
    try:
        import imageio.v2 as iio
        rd = iio.get_reader(src_mp4)
        meta = rd.get_meta_data()
        src_fps = float(meta.get("fps", 24.0) or 24.0)
        frames = [f for f in rd]
        rd.close()
        T = len(frames)
        if T == 0:
            return None
        step = max(1, int(round(src_fps / float(target_fps))))
        idxs = list(range(0, T, step)) or [0]
        os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
        wr = iio.get_writer(out_mp4, fps=target_fps, quality=SAMPLE_QUALITY, macro_block_size=1)
        for i in idxs:
            wr.append_data(frames[i])
        wr.close()
        return out_mp4 if (os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 2000) else None
    except Exception:
        try:
            if os.path.exists(out_mp4):
                os.remove(out_mp4)
        except Exception:
            pass
        return None

# ---- The two prompts from paper Fig.9 (aligned with the original wording) ----
Q_DESCRIBE = (
    "Describe this video in detail, ensuring you cover the main subjects, scene, and background. "
    "Focus on transformations throughout the video, including the motion, action, and any changes "
    "of the dynamic elements over time."
)
Q_IDENTIFY = (
    "Based on the video and the description, identify the most prominent dynamic object in the video, "
    "such as an animal, a person, or a vehicle that has clear independent motion. "
    "Answer with a set of nouns (lowercase, singular, concrete object names), separated with commas. "
    "Return exactly NULL if there is no significant dynamic object. "
    "Output ONLY the nouns or NULL, no explanation."
)


def out_paths(out_dir):
    label_dir = os.path.join(out_dir, "_labels")
    return label_dir


def label_path(out_dir, stem):
    return os.path.join(out_dir, "_labels", f"{stem}.json")


def _qwen_video(messages_content, timeout=TIMEOUT):
    """Call Qwen VL with a content list (video/text); return text, or None on failure."""
    payload = {
        "model": QWEN_VL_MODEL,
        "messages": [{"role": "user", "content": messages_content}],
        "max_tokens": 600,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(QWEN_VL_URL, json=payload, timeout=timeout)
        if r.status_code != 200:
            return None
        msg = r.json()["choices"][0]["message"]
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()
    except Exception:
        return None


def _parse_nouns(txt):
    """Parse Qwen's noun answer into a list. NULL/empty -> []."""
    if not txt:
        return []
    t = txt.strip()
    # strip possible quotes / trailing punctuation / prefixes
    t = re.sub(r'^(nouns?|answer|objects?)\s*[:：]\s*', '', t, flags=re.I).strip()
    if re.search(r'\bNULL\b', t, flags=re.I) or t.upper() == "NULL":
        return []
    # split on comma / Chinese comma / semicolon
    parts = re.split(r'[,，、;；]', t)
    nouns = []
    for p in parts:
        p = p.strip().strip('.。"\'').strip().lower()
        # drop obvious full-sentence (non-noun) answers
        if p and len(p) <= 40 and "null" not in p:
            nouns.append(p)
    return nouns


def label_one(video_path, out_dir, force=False):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    p = label_path(out_dir, stem)
    if not force and os.path.exists(p):
        try:
            c = json.load(open(p))
            if c.get("label_version") == LABEL_VERSION and "skip" in c:
                return c
        except Exception:
            pass

    # downsampled small clip to feed Qwen
    sampled_dir = os.path.join(out_dir, "_labels", "_sampled")
    os.makedirs(sampled_dir, exist_ok=True)
    sampled = os.path.join(sampled_dir, f"{stem}_{LABEL_FPS}fps.mp4")
    sampled = make_fps_sampled(video_path, sampled, target_fps=LABEL_FPS)
    if not sampled:
        rec = {"stem": stem, "description": "", "nouns": [], "primary_noun": "",
               "skip": True, "reason": "sample_failed", "label_version": LABEL_VERSION}
        _save(p, rec); return rec

    desc = nouns_txt = None
    for _ in range(MAX_RETRIES):
        # Q1 describe
        desc = _qwen_video([
            {"type": "text", "text": Q_DESCRIBE},
            {"type": "video_url", "video_url": {"url": f"file://{sampled}"}},
        ])
        if desc:
            break
    if not desc:
        # describe failed: do not write a terminal record (leave for retry)
        return {"stem": stem, "skip": None, "reason": "describe_failed", "label_version": LABEL_VERSION}

    for _ in range(MAX_RETRIES):
        # Q2 identify (with description + video)
        nouns_txt = _qwen_video([
            {"type": "text", "text": f"Video description: {desc}\n\n{Q_IDENTIFY}"},
            {"type": "video_url", "video_url": {"url": f"file://{sampled}"}},
        ])
        if nouns_txt is not None:
            break
    if nouns_txt is None:
        return {"stem": stem, "skip": None, "reason": "identify_failed", "label_version": LABEL_VERSION}

    nouns = _parse_nouns(nouns_txt)
    rec = {
        "stem": stem,
        "description": desc[:1200],
        "nouns": nouns,
        "primary_noun": nouns[0] if nouns else "",   # take the first of multiple objects (paper Filtering rule (2))
        "nouns_raw": nouns_txt[:200],
        "skip": (len(nouns) == 0),                    # no significant object -> skip (rule (1))
        "reason": "no_dynamic_object" if not nouns else "ok",
        "label_version": LABEL_VERSION,
    }
    _save(p, rec)
    return rec


def _save(p, rec):
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def collect_videos(video_dir, limit=0):
    import glob
    vids = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if limit > 0:
        vids = vids[:limit]
    return vids


def run_label(video_dir, out_dir, limit=0, workers=8, force=False):
    vids = collect_videos(video_dir, limit)
    print(f"[Label] videos to label: {len(vids)}  endpoint={QWEN_VL_URL} model={QWEN_VL_MODEL} fps={LABEL_FPS}", flush=True)
    recs = {}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(label_one, v, out_dir, force): v for v in vids}
        for fut in cf.as_completed(futs):
            v = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"stem": os.path.splitext(os.path.basename(v))[0], "skip": None,
                     "reason": f"exc:{e}", "label_version": LABEL_VERSION}
            recs[r["stem"]] = r
            done += 1
            if done % 20 == 0 or done == len(vids):
                kept = sum(1 for x in recs.values() if x.get("skip") is False)
                print(f"  Label progress {done}/{len(vids)}  with-object(keep)={kept}", flush=True)
    keep = sum(1 for r in recs.values() if r.get("skip") is False)
    null = sum(1 for r in recs.values() if r.get("skip") is True)
    err = sum(1 for r in recs.values() if r.get("skip") is None)
    print(f"[Label] done: keep={keep} skip(NULL)={null} ERR/unjudged={err} / total {len(recs)}", flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser(description="VideoComp Labeling (Qwen VL watches video -> most prominent dynamic-object nouns)")
    ap.add_argument("--video_dir", required=True, help="source video directory")
    ap.add_argument("--out_dir", default=_DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    recs = run_label(args.video_dir, args.out_dir, args.limit, args.workers, args.force)
    print("\n===== labeling samples =====")
    for stem, r in list(recs.items())[:10]:
        tag = "KEEP" if r.get("skip") is False else ("NULL" if r.get("skip") is True else "ERR")
        print(f"  [{tag}] {stem}: nouns={r.get('nouns')} primary={r.get('primary_noun','')}")


if __name__ == "__main__":
    main()
