#!/usr/bin/env python3
"""
Run GR00T N1.5 on the original GR1 dataset (HF file-repo) and dump teacher latents + pointers.

Key behaviors:
- Enumerates only top-level GR1 task dirs (names starting with "gr1_") via HfFileSystem.ls (no deep crawl).
- For each task, streams meta/episodes.jsonl to get episode indices and (optional) episode_chunk.
- If episode_chunk is missing, probes only that task's data/ and videos/ chunk folders to find the right shard.
- Constructs exact parquet/mp4 relative paths from each task's meta/info.json templates, then prefixes with the task dir.
- Downloads only the exact parquet/mp4 files we touch.
- Builds per-step inputs (state slices from modality.json + video frame), runs GR00T policy once per step, and captures
  pre-decoder latents via a forward hook on the DiT/transformer stack.
- Writes per-episode latents as .npz and a compact JSONL manifest with pointers (parquet path, row id, video path, etc.).

Notes:
* Uses the correct HfFileSystem URI scheme: "hf://datasets/{repo_id}/{path}".
* Per-task meta is authoritative (we don't assume a global "gr1_arms_only" dir exists).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Optional video backends
try:
    import decord  # type: ignore
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

try:
    import cv2  # type: ignore
    _HAS_OPENCV = True
except Exception:
    _HAS_OPENCV = False

try:
    import av  # type: ignore
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import torch  # type: ignore

from huggingface_hub import (
    HfFileSystem,
    hf_hub_download,
)

# ---------------------------
# Dataset repo interaction
# ---------------------------

def list_gr1_tasks(repo_id: str, *, repo_type: str = "dataset") -> List[str]:
    """List only top-level directories starting with 'gr1_' using HF FS (stable across hub versions)."""
    fs = HfFileSystem()
    items = fs.ls(f"hf://{repo_type}s/{repo_id}", detail=True)
    tasks: List[str] = []
    for it in items:
        if isinstance(it, dict) and it.get("type") == "directory":
            name = Path(it["name"]).name
            if name.startswith("gr1_"):
                tasks.append(name)
    return sorted(tasks)


def _hf_open(fs: HfFileSystem, repo_id: str, relpath: str, *, repo_type: str = "dataset"):
    """Open a repo file using the scheme 'hf://{repo_type}s/{repo_id}/{relpath}'."""
    uri = f"hf://{repo_type}s/{repo_id}/{relpath}"
    return fs.open(uri, "rb")


def read_json_from_repo(repo_id: str, relpath: str, *, repo_type: str = "dataset") -> dict:
    fs = HfFileSystem()
    with _hf_open(fs, repo_id, relpath, repo_type=repo_type) as f:
        return json.loads(f.read().decode("utf-8"))


def iter_jsonl_from_repo(repo_id: str, relpath: str, *, repo_type: str = "dataset") -> Iterator[dict]:
    fs = HfFileSystem()
    with _hf_open(fs, repo_id, relpath, repo_type=repo_type) as f:
        for line in f:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            yield json.loads(line)


def hf_download(repo_id: str, relpath: str, *, repo_type: str = "dataset") -> str:
    """Download a single file to the local HF cache and return the local path."""
    return hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=relpath)


# ---------------------------
# Video frame extraction
# ---------------------------

class VideoFrameReader:
    def __init__(self, video_path: str, fps_hint: Optional[float] = None):
        self.video_path = video_path
        self.fps_hint = fps_hint
        self._length = None
        self._fps = None

        if _HAS_DECORD:
            try:
                import decord
                # Your environment supports: native/mxnet/torch/tensorflow/tvm; use native
                decord.bridge.set_bridge("native")
            except Exception:
                pass
            self._vr = decord.VideoReader(video_path)
            self._length = len(self._vr)
            try:
                self._fps = float(self._vr.get_avg_fps())
            except Exception:
                self._fps = fps_hint
        elif _HAS_OPENCV:
            self._cap = cv2.VideoCapture(video_path)
            self._length = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._fps = float(fps) if fps and fps > 0 else fps_hint
        elif _HAS_PYAV:
            self._container = av.open(video_path)
            stream = self._container.streams.video[0]
            self._fps = float(stream.average_rate) if stream.average_rate else fps_hint
            self._length = stream.frames if stream.frames and stream.frames > 0 else None
        else:
            raise RuntimeError("No video backend found. Install decord or opencv-python or PyAV.")

    @property
    def fps(self) -> Optional[float]:
        return self._fps

    @property
    def length(self) -> Optional[int]:
        return self._length

    def get_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(0, frame_idx)
        if self._length is not None:
            frame_idx = min(frame_idx, self._length - 1)

        if _HAS_DECORD:
            img = self._vr[frame_idx]
            return img
        elif _HAS_OPENCV:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, (self._length or 1) - 1))
                ok, frame_bgr = self._cap.read()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return frame_rgb
        elif _HAS_PYAV:
            target = frame_idx
            cur = 0
            last = None
            for frame in self._container.decode(video=0):
                last = frame
                if cur == target:
                    return frame.to_ndarray(format="rgb24")
                cur += 1
            return last.to_ndarray(format="rgb24") if last is not None else np.zeros((1,1,3), dtype=np.uint8)
        else:
            raise RuntimeError("No video backend available.")


# ---------------------------
# GR00T policy + latent hook
# ---------------------------

def build_policy(model_path: str, embodiment_tag: str):
    from gr00t.model.policy import Gr00tPolicy  # type: ignore
    from gr00t.experiment.data_config import DATA_CONFIG_MAP  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    return policy


@dataclass
class LatentHook:
    handle: Optional[torch.utils.hooks.RemovableHandle]
    path: str
    latest: Optional[torch.Tensor] = None


def _pick_latent_module(model: torch.nn.Module, preferred: Optional[str] = None) -> Tuple[str, torch.nn.Module]:
    candidates: List[Tuple[str, torch.nn.Module]] = []
    for name, mod in model.named_modules():
        lname = name.lower()
        if preferred and name == preferred:
            return name, mod
        if any(key in lname for key in ("dit", "mmd", "transformer")):
            candidates.append((name, mod))
    if not candidates:
        return "model", model
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    return candidates[0]


def attach_latent_hook(model: torch.nn.Module, preferred: Optional[str] = None) -> LatentHook:
    name, module = _pick_latent_module(model, preferred=preferred)

    def _hook(_mod, _inp, out):
        tensor: Optional[torch.Tensor] = None
        if isinstance(out, torch.Tensor):
            tensor = out
        elif isinstance(out, (list, tuple)) and out:
            for item in out:
                if isinstance(item, torch.Tensor):
                    tensor = item
                    break
        if tensor is not None:
            hook.latest = tensor.detach().to("cpu")
        else:
            hook.latest = None

    hook = LatentHook(handle=None, path=name, latest=None)
    handle = module.register_forward_hook(_hook)
    hook.handle = handle
    return hook


# ---------------------------
# Episode iteration + packing
# ---------------------------

def make_paths(info_json: dict, modality_json: dict, episode_index: int, episode_chunk: int) -> Tuple[str, str]:
    data_tpl = info_json["data_path"]
    video_tpl = info_json["video_path"]
    ego_key = modality_json.get("video", {}).get("ego_view", {}).get("original_key", "observation.images.ego_view")
    parquet_rel = data_tpl.format(episode_chunk=episode_chunk, episode_index=episode_index)
    video_rel = video_tpl.format(episode_chunk=episode_chunk, episode_index=episode_index, video_key=ego_key)
    return parquet_rel, video_rel


def slice_state_action(state_vec: np.ndarray, action_vec: Optional[np.ndarray], modality_json: dict) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for space in ("state", "action"):
        vec = state_vec if space == "state" else action_vec
        if vec is None:
            continue
        for part, rng in modality_json.get(space, {}).items():
            s = rng["start"]
            e = rng["end"]
            key = f"{space}.{part}"
            out[key] = np.asarray(vec[s:e], dtype=np.float32)[None, ...]
    return out


def load_parquet_table(repo_id: str, relpath: str) -> pq.ParquetFile:
    fs = HfFileSystem()
    with _hf_open(fs, repo_id, relpath, repo_type="dataset") as f:
        data = f.read()
    reader = pa.BufferReader(data)
    return pq.ParquetFile(reader)


def frame_index_from_timestamp(ts: float, fps: float) -> int:
    return int(round(ts * fps))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# --------- chunk discovery helpers (when episode_chunk is missing) ----------

def find_chunk_for_parquet(repo_id: str, task: str, info_json: dict, episode_index: int) -> Optional[int]:
    """List task/data and find which chunk dir actually contains episode_<idx>.parquet."""
    fs = HfFileSystem()
    base = f"hf://datasets/{repo_id}/{task}/data"
    try:
        items = fs.ls(base, detail=True)
    except Exception:
        return None
    chunk_dirs = sorted(
        Path(it["name"]).name for it in items
        if isinstance(it, dict) and it.get("type") == "directory" and Path(it["name"]).name.startswith("chunk-")
    )
    for chunk_name in chunk_dirs:
        try:
            chunk_num = int(chunk_name.split("-")[1])
        except Exception:
            continue
        data_tpl = info_json["data_path"]
        candidate_rel = data_tpl.format(episode_chunk=chunk_num, episode_index=episode_index)
        try:
            HfFileSystem().info(f"hf://datasets/{repo_id}/{task}/{candidate_rel}")
            return chunk_num
        except Exception:
            continue
    return None


def find_chunk_for_video(repo_id: str, task: str, info_json: dict, modality_json: dict, episode_index: int) -> Optional[int]:
    """Same strategy but under videos/; returns chunk that contains the episode_<idx>.mp4."""
    fs = HfFileSystem()
    base = f"hf://datasets/{repo_id}/{task}/videos"
    try:
        items = fs.ls(base, detail=True)
    except Exception:
        return None
    chunk_dirs = sorted(
        Path(it["name"]).name for it in items
        if isinstance(it, dict) and it.get("type") == "directory" and Path(it["name"]).name.startswith("chunk-")
    )
    ego_key = modality_json.get("video", {}).get("ego_view", {}).get("original_key", "observation.images.ego_view")
    for chunk_name in chunk_dirs:
        try:
            chunk_num = int(chunk_name.split("-")[1])
        except Exception:
            continue
        video_tpl = info_json["video_path"]
        candidate_rel = video_tpl.format(episode_chunk=chunk_num, episode_index=episode_index, video_key=ego_key)
        try:
            HfFileSystem().info(f"hf://datasets/{repo_id}/{task}/{candidate_rel}")
            return chunk_num
        except Exception:
            continue
    return None


# ---------------------------
# Main runner
# ---------------------------

def run(
    repo_id: str = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
    model_path: str = "nvidia/GR00T-N1.5-3B",
    embodiment_tag: str = "gr1",
    output_dir: str = "distill_out",
    tasks_filter: Optional[str] = None,
    max_episodes_per_task: Optional[int] = None,
    start_at_episode: int = 0,
    preferred_latent_module: Optional[str] = None,
    require_video: bool = True,
    save_every: int = 1,
):
    t0 = time.time()

    out_root = Path(output_dir)
    ensure_dir(out_root)

    # Optional default fps if a task meta lacks it
    fps_default = 20.0

    print("[*] Loading GR00T policy...")
    policy = build_policy(model_path, embodiment_tag)
    hook = attach_latent_hook(policy.model, preferred=preferred_latent_module)
    print(f"[*] Latent hook attached at: {hook.path}")

    # Enumerate tasks
    task_dirs = list_gr1_tasks(repo_id)
    if tasks_filter:
        rx = re.compile(tasks_filter)
        task_dirs = [t for t in task_dirs if rx.search(Path(t).name)]
    print(f"[*] Found {len(task_dirs)} GR1 task dirs after filter.")

    # Optional global mapping from task-desc id -> text (best effort; not required)
    task_text_map: Dict[int, str] = {}
    try:
        # If present in a canonical task folder; ignore errors
        for rec in iter_jsonl_from_repo(repo_id, "gr1_arms_only/meta/tasks.jsonl"):
            _id = rec.get("id") or rec.get("task_id") or rec.get("index") or rec.get("task_index")
            _tx = rec.get("text") or rec.get("description") or rec.get("name") or rec.get("task")
            if _id is not None and _tx is not None:
                task_text_map[int(_id)] = str(_tx)
    except Exception:
        pass

    # Per-task processing
    for task in task_dirs:
        print(f"\n=== Task: {task} ===")

        # Read per-task meta
        try:
            info = read_json_from_repo(repo_id, f"{task}/meta/info.json")
            modality = read_json_from_repo(repo_id, f"{task}/meta/modality.json")
        except Exception as e:
            print(f"  ! Missing meta for {task}: {e}")
            continue

        fps = float(info.get("fps", fps_default))

        # Prepare output dirs
        out_task = out_root / Path(task).name
        out_lat = out_task / "latents"
        out_meta = out_task / "metadata"
        ensure_dir(out_lat)
        ensure_dir(out_meta)

        # Episodes listing
        try:
            episodes_iter = iter_jsonl_from_repo(repo_id, f"{task}/meta/episodes.jsonl")
        except Exception as e:
            print(f"  ! No episodes.jsonl for {task}: {e}")
            continue

        ep_count = 0
        for rec in episodes_iter:
            # Episode index and optional chunk
            try:
                episode_index = int(rec.get("episode_index"))
            except Exception:
                continue
            if episode_index < start_at_episode:
                continue

            raw_chunk = rec.get("episode_chunk", None)
            if raw_chunk is not None:
                episode_chunk = int(raw_chunk)
            else:
                epc_data = find_chunk_for_parquet(repo_id, task, info, episode_index)
                epc_vid  = find_chunk_for_video(repo_id, task, info, modality, episode_index)
                if epc_data is not None:
                    episode_chunk = epc_data
                elif epc_vid is not None:
                    episode_chunk = epc_vid
                else:
                    episode_chunk = 0  # final fallback

            # Build task-relative paths for HF
            parquet_rel, video_rel = make_paths(info, modality, episode_index, episode_chunk)
            parquet_rel_task = f"{task}/{parquet_rel}"
            video_rel_task   = f"{task}/{video_rel}"

            # Per-task cap
            if max_episodes_per_task is not None and ep_count >= max_episodes_per_task:
                break

            # Skip if already processed
            lat_out = out_lat / f"episode_{episode_index:06d}.npz"
            meta_out = out_meta / f"episode_{episode_index:06d}.jsonl"
            if lat_out.exists() and meta_out.exists():
                print(f"  - Episode {episode_index:06d}: already done; skipping.")
                ep_count += 1
                continue

            # Load parquet
            try:
                pf = load_parquet_table(repo_id, parquet_rel_task)
            except Exception as e:
                print(f"  ! Failed to load parquet {parquet_rel_task}: {e}")
                continue

            # Prepare video (optional)
            vr = None
            if require_video:
                try:
                    video_local = hf_download(repo_id, video_rel_task)
                    vr = VideoFrameReader(video_local, fps_hint=fps)
                except Exception as e:
                    print(f"  ! Failed to prepare video {video_rel_task}: {e}")
                    print("    -> Skipping episode (video required).")
                    continue
            else:
                # Best-effort video if available (won't block)
                try:
                    video_local = hf_download(repo_id, video_rel_task)
                    vr = VideoFrameReader(video_local, fps_hint=fps)
                except Exception:
                    vr = None

            # Read episode table
            table = pf.read()
            cols = {name: table[name].to_numpy() for name in table.column_names}

            state_col = cols.get("observation.state")
            action_col = cols.get("action")
            ts_col = cols.get("timestamp")
            task_idx_col = cols.get("task_index")
            desc_col = cols.get("annotation.human.action.task_description")
            valid_col = cols.get("annotation.human.validity")

            latents_per_step: List[np.ndarray] = []
            manifest_lines: List[str] = []

            n_rows = len(table)
            for row_id in range(n_rows):
                # Build model input for this step
                state_vec = np.array(state_col[row_id], dtype=np.float32) if state_col is not None else None
                action_vec = np.array(action_col[row_id], dtype=np.float32) if action_col is not None else None
                ts = float(ts_col[row_id]) if ts_col is not None else float(row_id) / fps

                if state_vec is None or state_vec.shape[0] < 44:
                    continue

                step = slice_state_action(state_vec, None, modality)

                # Try to attach the real frame (if we have a video reader)
                frame_idx = frame_index_from_timestamp(ts, vr.fps if (vr and vr.fps) else fps)
                if vr is not None:
                    try:
                        img = vr.get_frame(frame_idx)
                        step["video.ego_view"] = img[None, ...]  # (1, H, W, 3)
                    except Exception as e:
                        if require_video:
                            print(f"    ! Failed to read frame {frame_idx} in {video_rel_task}: {e}")
                            continue
                        # else: leave missing for now; add dummy below

                # Ensure the video key exists for the transform
                if "video.ego_view" not in step:
                    if require_video:
                        continue  # strict mode
                    else:
                        # Provide a dummy frame so VideoToTensor doesn't crash
                        try:
                            if vr is not None:
                                _probe = vr.get_frame(0)
                                _h, _w = int(_probe.shape[0]), int(_probe.shape[1])
                            else:
                                _h, _w = 256, 256
                        except Exception:
                            _h, _w = 256, 256
                        step["video.ego_view"] = np.zeros((1, _h, _w, 3), dtype=np.uint8)

                # Optional annotations
                if desc_col is not None:
                    try:
                        desc_id = int(desc_col[row_id])
                        if desc_id in task_text_map:
                            step["annotation.human.action.task_description"] = np.array([desc_id], dtype=np.int64)
                    except Exception:
                        pass

                if task_idx_col is not None:
                    try:
                        step["task_index"] = np.array([int(task_idx_col[row_id])], dtype=np.int64)
                    except Exception:
                        pass

                if valid_col is not None:
                    try:
                        step["annotation.human.validity"] = np.array([int(valid_col[row_id])], dtype=np.int64)
                    except Exception:
                        pass

                # Inference
                with torch.no_grad():
                    _ = policy.get_action(step)

                if not isinstance(getattr(hook, "latest", None), torch.Tensor):
                    continue

                lat = hook.latest
                latents_per_step.append(lat.to(dtype=torch.float16).cpu().numpy())

                # Pointer record for distillation
                ptr = {
                    "parquet_path": parquet_rel_task,
                    "row_id": int(row_id),
                    "timestamp": ts,
                    "video_path": video_rel_task,
                    "video_frame_index": int(frame_idx),
                    "episode_index": episode_index,
                    "episode_chunk": episode_chunk,
                    "task_index": int(task_idx_col[row_id]) if task_idx_col is not None else None,
                    "task_desc_id": int(desc_col[row_id]) if desc_col is not None else None,
                    "valid": int(valid_col[row_id]) if valid_col is not None else None,
                    "latent_module": hook.path,
                }
                manifest_lines.append(json.dumps(ptr))

            if not latents_per_step:
                print(f"  - Episode {episode_index:06d}: no latents captured; skipping save.")
                continue

            # Save per-episode outputs
            same_shape = len({tuple(x.shape) for x in latents_per_step}) == 1
            lat_out = out_lat / f"episode_{episode_index:06d}.npz"
            meta_out = out_meta / f"episode_{episode_index:06d}.jsonl"

            if same_shape:
                lat_arr = np.stack(latents_per_step, axis=0)  # (T, ...)
                np.savez_compressed(lat_out, latents=lat_arr, module=hook.path, dtype=str(lat_arr.dtype))
            else:
                obj = np.empty((len(latents_per_step),), dtype=object)
                for i, arr in enumerate(latents_per_step):
                    obj[i] = arr
                np.savez_compressed(lat_out, latents=obj, module=hook.path, ragged=True)

            with open(meta_out, "w", encoding="utf-8") as f:
                for line in manifest_lines:
                    f.write(line + "\n")

            print(f"  + Episode {episode_index:06d}: wrote latents -> {lat_out.name}  | meta -> {meta_out.name}")
            ep_count += 1

    if hook.handle is not None:
        hook.handle.remove()
    print(f"\n[*] Done in {time.time() - t0:.1f}s. Output at: {out_root.resolve()}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GR00T N1.5 over GR1 episodes and dump teacher latents.")
    p.add_argument("--repo", default="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim", help="HF dataset repo id")
    p.add_argument("--model", default="nvidia/GR00T-N1.5-3B", help="Model repo id or local path")
    p.add_argument("--embodiment", default="gr1", help="Embodiment tag for the policy")
    p.add_argument("--out", default="distill_out", help="Output directory")
    p.add_argument("--tasks-filter", default=None, help="Regex to filter task dir names (e.g., 'arms_only|CanSort')")
    p.add_argument("--max-episodes-per-task", type=int, default=None, help="Optional cap per task")
    p.add_argument("--start-at-episode", type=int, default=0, help="Skip episodes < this index")
    p.add_argument("--latent-module", default=None, help="Preferred module path for forward hook (if known)")
    p.add_argument("--require-video", action="store_true", help="Require video frames (skip episode if video missing)")
    p.add_argument("--save-every", type=int, default=1, help="Flush cadence (reserved)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        repo_id=args.repo,
        model_path=args.model,
        embodiment_tag=args.embodiment,
        output_dir=args.out,
        tasks_filter=args.tasks_filter,
        max_episodes_per_task=args.max_episodes_per_task,
        start_at_episode=args.start_at_episode,
        preferred_latent_module=args.latent_module,
        require_video=args.require_video,
        save_every=args.save_every,
    )
