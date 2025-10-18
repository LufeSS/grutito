#!/usr/bin/env python3
"""
Run GR00T N1.5 on the original GR1 dataset (HF file-repo) and dump teacher latents + pointers.

This version captures the **input to the decoder** by attaching a forward_pre_hook on the decoder
module (so we record the tensor right before decoding). We then drop the singleton batch dimension
and slice the trailing action tokens, yielding exactly the latent vectors the decoder consumes when
predicting the action trajectory. This avoids relying on internal DiT layer names and gives you the
final DiT output that the decoder actually uses.

Other key behaviors (unchanged):
- Enumerate only top-level GR1 task dirs via HfFileSystem.ls (no deep crawl).
- For each task, stream meta/episodes.jsonl to get episode_index and (optional) episode_chunk.
- If episode_chunk is missing, probe only that task's data/ and videos/ chunk folders.
- Construct exact parquet/mp4 paths using per-task meta/info.json templates.
- Download only the exact parquet/mp4 we touch (HF cache).
- Build step-wise inputs (state slices + video frame), run policy, and store latents.
- Write per-episode .npz latents and a JSONL manifest of pointers including both the
  original (pre-slice) sequence shape and the post-slice decoder-action dimensions.

Notes:
* Uses the correct HfFileSystem URI scheme: "hf://datasets/{repo_id}/{path}".
* Per-task meta is authoritative.
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
                decord.bridge.set_bridge("numpy")
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
            try:
                return img.asnumpy()
            except AttributeError:
                return np.asarray(img)
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
# GR00T policy + hooks
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


from dataclasses import dataclass
@dataclass
class LatentHook:
    handle: Optional[torch.utils.hooks.RemovableHandle]
    path: str
    latest: Optional[torch.Tensor] = None
    kind: str = "forward_pre"


def _find_module_by_name(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    for n, m in model.named_modules():
        if n == name:
            return m
    return None


def attach_decoder_input_hook(model: torch.nn.Module, preferred: Optional[str] = None) -> LatentHook:
    target_name = None
    target_module = None

    if preferred:
        target_module = _find_module_by_name(model, preferred)
        target_name = preferred if target_module is not None else None

    if target_module is None:
        candidates: List[Tuple[str, torch.nn.Module]] = []
        for name, mod in model.named_modules():
            lname = name.lower()
            cls = mod.__class__.__name__.lower()
            if ("decoder" in lname or "decoder" in cls) and ("action" in lname or "head" in lname or "policy" in lname):
                candidates.append((name, mod))
        if not candidates:
            for name, mod in model.named_modules():
                lname = name.lower()
                cls = mod.__class__.__name__.lower()
                if ("decoder" in lname or "decoder" in cls):
                    candidates.append((name, mod))
        if candidates:
            candidates.sort(key=lambda x: len(x[0]), reverse=True)
            target_name, target_module = candidates[0]

    if target_module is None:
        for name, mod in model.named_modules():
            if name.endswith("action_head"):
                target_name, target_module = name, mod
                break

    if target_module is None:
        target_name, target_module = "model", model

    hook = LatentHook(handle=None, path=target_name or "unknown", latest=None, kind="forward_pre")

    def _pre_hook(_mod, inputs):
        ten = None
        for x in inputs:
            if isinstance(x, torch.Tensor):
                ten = x
                break
            if isinstance(x, (list, tuple)):
                for y in x:
                    if isinstance(y, torch.Tensor):
                        ten = y
                        break
            if ten is not None:
                break
        hook.latest = ten.detach().to("cpu") if isinstance(ten, torch.Tensor) else None

    handle = target_module.register_forward_pre_hook(_pre_hook, with_kwargs=False)
    hook.handle = handle
    return hook


# ---------------------------
# Episode iteration + packing
# ---------------------------

def get_video_key_candidates(modality_json: dict, override: Optional[str] = None) -> List[str]:
    candidates: List[str] = []
    if override:
        candidates.append(override)

    front_view = modality_json.get("video", {}).get("front_view", {})
    for field in ("original_key", "key", "path"):
        val = front_view.get(field)
        if isinstance(val, str):
            candidates.append(val)

    # Provide sensible fallbacks.
    candidates.append("observation.images.front_view")
    candidates.append("observation.images.ego_view")

    seen = set()
    ordered: List[str] = []
    for cand in candidates:
        if not cand or cand in seen:
            continue
        ordered.append(cand)
        seen.add(cand)
    return ordered


def resolve_video_key_for_task(
    repo_id: str,
    task: str,
    info_json: dict,
    modality_json: dict,
    override: Optional[str] = None,
) -> str:
    candidates = get_video_key_candidates(modality_json, override=override)
    if not candidates:
        return "observation.images.front_view"

    fs = HfFileSystem()
    base = f"hf://datasets/{repo_id}/{task}/videos"

    try:
        items = fs.ls(base, detail=True)
    except Exception:
        return candidates[0]

    chunk_dirs = [
        Path(it["name"]).name
        for it in items
        if isinstance(it, dict) and it.get("type") == "directory"
    ]

    for cand in candidates:
        for chunk_name in chunk_dirs:
            try:
                children = fs.ls(f"{base}/{chunk_name}", detail=True)
            except Exception:
                continue
            dir_names = {
                Path(child["name"]).name
                for child in children
                if isinstance(child, dict) and child.get("type") == "directory"
            }
            if cand in dir_names:
                return cand

    return candidates[0]


def make_paths(info_json: dict, episode_index: int, episode_chunk: int, video_key: str) -> Tuple[str, str]:
    data_tpl = info_json["data_path"]
    video_tpl = info_json["video_path"]
    parquet_rel = data_tpl.format(episode_chunk=episode_chunk, episode_index=episode_index)
    video_rel = video_tpl.format(episode_chunk=episode_chunk, episode_index=episode_index, video_key=video_key)
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


def find_chunk_for_parquet(repo_id: str, task: str, info_json: dict, episode_index: int) -> Optional[int]:
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


def find_chunk_for_video(
    repo_id: str,
    task: str,
    info_json: dict,
    episode_index: int,
    video_keys: List[str],
) -> Tuple[Optional[int], Optional[str]]:
    fs = HfFileSystem()
    base = f"hf://datasets/{repo_id}/{task}/videos"
    try:
        items = fs.ls(base, detail=True)
    except Exception:
        return None, None
    chunk_dirs = sorted(
        Path(it["name"]).name for it in items
        if isinstance(it, dict) and it.get("type") == "directory" and Path(it["name"]).name.startswith("chunk-")
    )
    for chunk_name in chunk_dirs:
        try:
            chunk_num = int(chunk_name.split("-")[1])
        except Exception:
            continue
        video_tpl = info_json["video_path"]
        for video_key in video_keys:
            candidate_rel = video_tpl.format(
                episode_chunk=chunk_num,
                episode_index=episode_index,
                video_key=video_key,
            )
            try:
                HfFileSystem().info(f"hf://datasets/{repo_id}/{task}/{candidate_rel}")
                return chunk_num, video_key
            except Exception:
                continue
    return None, None


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
    decoder_module: Optional[str] = None,
    require_video: bool = True,
    video_key_override: Optional[str] = None,
    save_every: int = 1,
):
    t0 = time.time()

    out_root = Path(output_dir)
    ensure_dir(out_root)

    fps_default = 20.0

    print("[*] Loading GR00T policy...")
    policy = build_policy(model_path, embodiment_tag)
    hook = attach_decoder_input_hook(policy.model, preferred=decoder_module)
    print(f"[*] Decoder-input hook attached (pre-hook) at: {hook.path}")

    # Give a one-time breakdown of how many tokens the decoder receives and which
    # subset corresponds to the action latents we ultimately care about.
    state_tokens = 0
    try:
        state_indices = policy.state_delta_indices
        if state_indices is not None:
            state_tokens = int(len(state_indices))
    except Exception:
        state_tokens = 0
    future_tokens = int(getattr(policy.model.action_head.config, "num_target_vision_tokens", 0) or 0)
    action_tokens = int(getattr(policy.model.action_head, "action_horizon", 0) or 0)
    total_tokens = state_tokens + future_tokens + action_tokens
    if total_tokens:
        print(
            "[*] Decoder expects sequence tokens: "
            f"state={state_tokens} + future={future_tokens} + action={action_tokens}"
            f" -> total={total_tokens}"
        )
        if action_tokens:
            print(
                "    Capturing DiT output before the decoder and slicing down to the"
                " last action tokens so the stored latents match the decoder's actual"
                " inputs for prediction."
            )

    task_dirs = list_gr1_tasks(repo_id)
    if tasks_filter:
        rx = re.compile(tasks_filter)
        task_dirs = [t for t in task_dirs if rx.search(Path(t).name)]
    print(f"[*] Found {len(task_dirs)} GR1 task dirs after filter.")

    task_text_map: Dict[int, str] = {}
    try:
        for rec in iter_jsonl_from_repo(repo_id, "gr1_arms_only/meta/tasks.jsonl"):
            _id = rec.get("id") or rec.get("task_id") or rec.get("index") or rec.get("task_index")
            _tx = rec.get("text") or rec.get("description") or rec.get("name") or rec.get("task")
            if _id is not None and _tx is not None:
                task_text_map[int(_id)] = str(_tx)
    except Exception:
        pass

    for task in task_dirs:
        print(f"\n=== Task: {task} ===")

        try:
            info = read_json_from_repo(repo_id, f"{task}/meta/info.json")
            modality = read_json_from_repo(repo_id, f"{task}/meta/modality.json")
        except Exception as e:
            print(f"  ! Missing meta for {task}: {e}")
            continue

        fps = float(info.get("fps", fps_default))

        out_task = out_root / Path(task).name
        out_lat = out_task / "latents"
        out_meta = out_task / "metadata"
        ensure_dir(out_lat)
        ensure_dir(out_meta)

        video_key_candidates = get_video_key_candidates(modality, override=video_key_override)
        initial_preference = video_key_candidates[0] if video_key_candidates else None
        resolved_video_key = resolve_video_key_for_task(
            repo_id,
            task,
            info,
            modality,
            override=video_key_override,
        )
        if resolved_video_key not in video_key_candidates:
            video_key_candidates.insert(0, resolved_video_key)
        video_key_order = [resolved_video_key] + [k for k in video_key_candidates if k != resolved_video_key]
        if not video_key_order:
            video_key_order = ["observation.images.front_view"]

        if video_key_override:
            print(f"  > Using video key override: {resolved_video_key}")
        elif initial_preference and resolved_video_key != initial_preference:
            print(
                "  > Video key '"
                f"{initial_preference}"
                "' unavailable; using '"
                f"{resolved_video_key}"
                "' instead."
            )

        try:
            episodes_iter = iter_jsonl_from_repo(repo_id, f"{task}/meta/episodes.jsonl")
        except Exception as e:
            print(f"  ! No episodes.jsonl for {task}: {e}")
            continue

        ep_count = 0
        for rec in episodes_iter:
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
                epc_vid, detected_key = find_chunk_for_video(
                    repo_id,
                    task,
                    info,
                    episode_index,
                    video_key_order,
                )
                if detected_key is not None and detected_key != video_key_order[0]:
                    video_key_order = [detected_key] + [k for k in video_key_order if k != detected_key]
                if epc_data is not None:
                    episode_chunk = epc_data
                elif epc_vid is not None:
                    episode_chunk = epc_vid
                else:
                    episode_chunk = 0

            if not video_key_order:
                video_key_order = ["observation.images.front_view"]

            first_key = video_key_order[0]
            parquet_rel, _ = make_paths(info, episode_index, episode_chunk, first_key)
            parquet_rel_task = f"{task}/{parquet_rel}"
            if max_episodes_per_task is not None and ep_count >= max_episodes_per_task:
                break

            lat_out = out_lat / f"episode_{episode_index:06d}.npz"
            meta_out = out_meta / f"episode_{episode_index:06d}.jsonl"
            if lat_out.exists() and meta_out.exists():
                print(f"  - Episode {episode_index:06d}: already done; skipping.")
                ep_count += 1
                continue

            try:
                pf = load_parquet_table(repo_id, parquet_rel_task)
            except Exception as e:
                print(f"  ! Failed to load parquet {parquet_rel_task}: {e}")
                continue

            video_rel = None
            video_rel_task = None
            video_download_error: Optional[Tuple[str, Exception]] = None
            vr = None

            for idx, candidate_key in enumerate(video_key_order):
                _, candidate_video_rel = make_paths(
                    info,
                    episode_index,
                    episode_chunk,
                    candidate_key,
                )
                candidate_task_rel = f"{task}/{candidate_video_rel}"
                video_rel = candidate_video_rel
                video_rel_task = candidate_task_rel
                try:
                    video_local = hf_download(repo_id, candidate_task_rel)
                    vr = VideoFrameReader(video_local, fps_hint=fps)
                    if idx != 0:
                        video_key_order = [candidate_key] + [k for k in video_key_order if k != candidate_key]
                    break
                except Exception as e:
                    video_download_error = (candidate_task_rel, e)
                    continue

            if video_rel is None:
                _, fallback_video_rel = make_paths(info, episode_index, episode_chunk, video_key_order[0])
                video_rel = fallback_video_rel
                video_rel_task = f"{task}/{fallback_video_rel}"

            if vr is None and require_video:
                if video_download_error is not None:
                    fail_path, fail_exc = video_download_error
                    print(f"  ! Failed to prepare video {fail_path}: {fail_exc}")
                else:
                    print(
                        f"  ! Failed to prepare video {video_rel_task}: unable to locate a valid video key"
                    )
                if len(video_key_order) > 1:
                    print("    -> Tried video keys: " + ", ".join(video_key_order))
                print("    -> Skipping episode (video required).")
                continue

            if vr is None and video_download_error is not None:
                video_rel_task = video_download_error[0]
                if video_rel_task.startswith(f"{task}/"):
                    video_rel = video_rel_task.split(f"{task}/", 1)[1]

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
                state_vec = np.array(state_col[row_id], dtype=np.float32) if state_col is not None else None
                action_vec = np.array(action_col[row_id], dtype=np.float32) if action_col is not None else None
                ts = float(ts_col[row_id]) if ts_col is not None else float(row_id) / fps

                if state_vec is None or state_vec.shape[0] < 44:
                    continue

                step = slice_state_action(state_vec, None, modality)

                frame_idx = frame_index_from_timestamp(ts, vr.fps if (vr and vr.fps) else fps)
                if vr is not None:
                    try:
                        img = vr.get_frame(frame_idx)
                        step["video.front_view"] = img[None, ...]
                    except Exception as e:
                        if require_video:
                            print(f"    ! Failed to read frame {frame_idx} in {video_rel_task}: {e}")
                            continue

                if "video.front_view" not in step:
                    if require_video:
                        continue
                    else:
                        try:
                            if vr is not None:
                                _probe = vr.get_frame(0)
                                _h, _w = int(_probe.shape[0]), int(_probe.shape[1])
                            else:
                                _h, _w = 256, 256
                        except Exception:
                            _h, _w = 256, 256
                        step["video.front_view"] = np.zeros((1, _h, _w, 3), dtype=np.uint8)

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

                with torch.no_grad():
                    _ = policy.get_action(step)

                if not isinstance(getattr(hook, "latest", None), torch.Tensor):
                    continue

                lat = hook.latest
                if not isinstance(lat, torch.Tensor):
                    continue

                # ``lat`` is the tensor fed to the action decoder and therefore has
                # shape (batch, seq_len, hidden_size).  We keep a copy of the full
                # sequence length for bookkeeping, but only the final ``action_horizon``
                # tokens influence the decoded action prediction.  By dropping the
                # singleton batch dimension and slicing to the trailing action tokens
                # we record exactly the latent vectors the decoder consumes when
                # producing actions.
                lat = lat.detach()
                original_shape = tuple(lat.shape)

                if lat.dim() == 3:
                    action_horizon = int(getattr(policy.model.action_head, "action_horizon", 0) or 0)
                    if action_horizon and lat.shape[1] >= action_horizon:
                        lat = lat[:, -action_horizon:, :]
                    if lat.shape[0] == 1:
                        lat = lat.squeeze(0)
                elif lat.dim() == 2:
                    action_horizon = int(getattr(policy.model.action_head, "action_horizon", 0) or 0)
                    if action_horizon and lat.shape[0] >= action_horizon:
                        lat = lat[-action_horizon:, :]

                lat_np = lat.to(dtype=torch.float16).contiguous().cpu().numpy()
                # Each saved array now has shape (action_horizon, hidden_size) (or
                # (batch, action_horizon, hidden_size) if a batch dimension is present),
                # so stacking across timesteps yields
                # (episode_len, action_horizon, hidden_size).
                latents_per_step.append(lat_np)

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
                    "hook_kind": hook.kind,
                    "latent_seq_len": original_shape[-2] if len(original_shape) >= 2 else None,
                    "latent_action_tokens": lat_np.shape[-2] if lat_np.ndim >= 2 else None,
                    "latent_hidden_size": lat_np.shape[-1] if lat_np.ndim >= 1 else None,
                    "latent_original_shape": list(original_shape),
                }
                manifest_lines.append(json.dumps(ptr))

            if not latents_per_step:
                print(f"  - Episode {episode_index:06d}: no latents captured; skipping save.")
                continue

            same_shape = len({tuple(x.shape) for x in latents_per_step}) == 1
            lat_out = out_lat / f"episode_{episode_index:06d}.npz"
            meta_out = out_meta / f"episode_{episode_index:06d}.jsonl"

            if same_shape:
                lat_arr = np.stack(latents_per_step, axis=0)
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
    p = argparse.ArgumentParser(description="Run GR00T N1.5 over GR1 episodes and dump teacher latents (decoder input)."
    )
    p.add_argument("--repo", default="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim", help="HF dataset repo id")
    p.add_argument("--model", default="nvidia/GR00T-N1.5-3B", help="Model repo id or local path")
    p.add_argument("--embodiment", default="gr1", help="Embodiment tag for the policy")
    p.add_argument("--out", default="distill_out", help="Output directory")
    p.add_argument("--tasks-filter", default=None, help="Regex to filter task dir names (e.g., 'arms_only|CanSort')")
    p.add_argument("--max-episodes-per-task", type=int, default=None, help="Optional cap per task")
    p.add_argument("--start-at-episode", type=int, default=0, help="Skip episodes < this index")
    p.add_argument("--decoder-module", default=None, help="Exact module path to attach pre-hook (optional)")
    p.add_argument("--require-video", action="store_true", help="Require video frames (skip episode if video missing)")
    p.add_argument(
        "--video-key",
        default=None,
        help="Override the video modality key (e.g., observation.images.front_view)",
    )
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
        decoder_module=args.decoder_module,
        require_video=args.require_video,
        video_key_override=args.video_key,
        save_every=args.save_every,
    )
