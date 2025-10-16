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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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


@dataclass
class EpisodeSample:
    episode_index: int
    episode_chunk: int
    parquet_rel_task: str
    video_rel_task: str
    columns: Dict[str, Optional[np.ndarray]]
    length: int
    fps: float
    video_reader: Optional[VideoFrameReader]
    latent_path: Path
    meta_path: Path


@dataclass
class EpisodeWorkItem:
    sample: EpisodeSample
    latents_per_step: List[np.ndarray]
    manifest_lines: List[str]


class EpisodeDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, samples: Sequence[EpisodeSample]):
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EpisodeSample:
        return self._samples[idx]


def collate_episode_steps(steps: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not steps:
        return {}
    common_keys = set(steps[0].keys())
    for step in steps[1:]:
        common_keys &= set(step.keys())
    batched: Dict[str, np.ndarray] = {}
    for key in sorted(common_keys):
        arrays = [step[key] for step in steps]
        if not arrays:
            continue
        try:
            batched[key] = np.concatenate(arrays, axis=0)
        except ValueError:
            batched[key] = np.stack(arrays, axis=0)
    return batched


def prepare_episode_step(
    sample: EpisodeSample,
    row_id: int,
    modality: dict,
    task_text_map: Dict[int, str],
    require_video: bool,
) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    state_col = sample.columns.get("observation.state")
    ts_col = sample.columns.get("timestamp")
    task_idx_col = sample.columns.get("task_index")
    desc_col = sample.columns.get("annotation.human.action.task_description")
    valid_col = sample.columns.get("annotation.human.validity")

    state_vec = (
        np.array(state_col[row_id], dtype=np.float32) if state_col is not None else None
    )
    ts = float(ts_col[row_id]) if ts_col is not None else float(row_id) / sample.fps

    if state_vec is None or state_vec.shape[0] < 44:
        return None

    step = slice_state_action(state_vec, None, modality)

    frame_idx = frame_index_from_timestamp(
        ts, sample.video_reader.fps if (sample.video_reader and sample.video_reader.fps) else sample.fps
    )
    if sample.video_reader is not None:
        try:
            img = sample.video_reader.get_frame(frame_idx)
            step["video.ego_view"] = img[None, ...]
        except Exception as e:
            if require_video:
                print(
                    f"    ! Failed to read frame {frame_idx} in {sample.video_rel_task}: {e}"
                )
                return None
    if "video.ego_view" not in step:
        if require_video:
            return None
        try:
            if sample.video_reader is not None:
                probe = sample.video_reader.get_frame(0)
                h, w = int(probe.shape[0]), int(probe.shape[1])
            else:
                h, w = 256, 256
        except Exception:
            h, w = 256, 256
        step["video.ego_view"] = np.zeros((1, h, w, 3), dtype=np.uint8)

    desc_id = None
    if desc_col is not None:
        try:
            desc_id = int(desc_col[row_id])
            if desc_id in task_text_map:
                step["annotation.human.action.task_description"] = np.array([desc_id], dtype=np.int64)
        except Exception:
            desc_id = None

    task_idx_val = None
    if task_idx_col is not None:
        try:
            task_idx_val = int(task_idx_col[row_id])
            step["task_index"] = np.array([task_idx_val], dtype=np.int64)
        except Exception:
            task_idx_val = None

    valid_val = None
    if valid_col is not None:
        try:
            valid_val = int(valid_col[row_id])
            step["annotation.human.validity"] = np.array([valid_val], dtype=np.int64)
        except Exception:
            valid_val = None

    meta = {
        "row_id": int(row_id),
        "timestamp": ts,
        "frame_idx": int(frame_idx),
        "task_index": task_idx_val,
        "task_desc_id": desc_id,
        "valid": valid_val,
    }

    return step, meta


def process_latent_tensor(policy, lat: torch.Tensor) -> Tuple[np.ndarray, Tuple[int, ...]]:
    lat = lat.detach()
    original_shape = tuple(lat.shape)

    action_horizon = int(getattr(policy.model.action_head, "action_horizon", 0) or 0)

    if lat.dim() == 3:
        if action_horizon and lat.shape[1] >= action_horizon:
            lat = lat[:, -action_horizon:, :]
    elif lat.dim() == 2:
        if action_horizon and lat.shape[0] >= action_horizon:
            lat = lat[-action_horizon:, :]

    lat_np = lat.to(dtype=torch.float16).contiguous().cpu().numpy()
    return lat_np, original_shape


def run_episode_batch(
    samples: Sequence[EpisodeSample],
    policy,
    hook: LatentHook,
    modality: dict,
    task_text_map: Dict[int, str],
    require_video: bool,
    batch_size: int,
) -> int:
    if not samples:
        return 0

    dataset = EpisodeDataset(samples)
    loader = torch.utils.data.DataLoader(  # type: ignore[attr-defined]
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    completed = 0

    for batch in loader:
        if not batch:
            continue
        work_items = [EpisodeWorkItem(sample=s, latents_per_step=[], manifest_lines=[]) for s in batch]
        max_rows = max(item.sample.length for item in work_items)

        for row_id in range(max_rows):
            steps: List[Dict[str, np.ndarray]] = []
            meta_records: List[Tuple[EpisodeWorkItem, Dict[str, Any]]] = []
            for item in work_items:
                if row_id >= item.sample.length:
                    continue
                prepared = prepare_episode_step(
                    item.sample,
                    row_id,
                    modality,
                    task_text_map,
                    require_video,
                )
                if prepared is None:
                    continue
                step_dict, meta = prepared
                steps.append(step_dict)
                meta_records.append((item, meta))

            if not steps:
                continue

            batched_inputs = collate_episode_steps(steps)
            if not batched_inputs:
                continue

            with torch.no_grad():
                _ = policy.get_action(batched_inputs)

            lat_tensor = getattr(hook, "latest", None)
            if not isinstance(lat_tensor, torch.Tensor):
                continue

            if lat_tensor.dim() >= 3:
                lat_splits = list(torch.unbind(lat_tensor, dim=0))
            elif lat_tensor.dim() == 2:
                lat_splits = [lat_tensor]
            else:
                lat_splits = []

            if len(lat_splits) != len(meta_records):
                lat_splits = lat_splits[: len(meta_records)]

            for (item, meta), lat_part in zip(meta_records, lat_splits):
                lat_np, original_shape = process_latent_tensor(policy, lat_part)
                item.latents_per_step.append(lat_np)
                ptr = {
                    "parquet_path": item.sample.parquet_rel_task,
                    "row_id": meta["row_id"],
                    "timestamp": meta["timestamp"],
                    "video_path": item.sample.video_rel_task,
                    "video_frame_index": meta["frame_idx"],
                    "episode_index": item.sample.episode_index,
                    "episode_chunk": item.sample.episode_chunk,
                    "task_index": meta.get("task_index"),
                    "task_desc_id": meta.get("task_desc_id"),
                    "valid": meta.get("valid"),
                    "latent_module": hook.path,
                    "hook_kind": hook.kind,
                    "latent_seq_len": original_shape[-2] if len(original_shape) >= 2 else None,
                    "latent_action_tokens": lat_np.shape[-2] if lat_np.ndim >= 2 else None,
                    "latent_hidden_size": lat_np.shape[-1] if lat_np.ndim >= 1 else None,
                    "latent_original_shape": list(original_shape),
                }
                item.manifest_lines.append(json.dumps(ptr))

        for item in work_items:
            if not item.latents_per_step:
                print(
                    f"  - Episode {item.sample.episode_index:06d}: no latents captured; skipping save."
                )
                continue

            same_shape = len({tuple(arr.shape) for arr in item.latents_per_step}) == 1

            if same_shape:
                lat_arr = np.stack(item.latents_per_step, axis=0)
                np.savez_compressed(
                    item.sample.latent_path,
                    latents=lat_arr,
                    module=hook.path,
                    dtype=str(lat_arr.dtype),
                )
            else:
                obj = np.empty((len(item.latents_per_step),), dtype=object)
                for i, arr in enumerate(item.latents_per_step):
                    obj[i] = arr
                np.savez_compressed(
                    item.sample.latent_path,
                    latents=obj,
                    module=hook.path,
                    ragged=True,
                )

            with open(item.sample.meta_path, "w", encoding="utf-8") as f:
                for line in item.manifest_lines:
                    f.write(line + "\n")

            print(
                "  + Episode "
                f"{item.sample.episode_index:06d}: wrote latents -> {item.sample.latent_path.name}"
                f"  | meta -> {item.sample.meta_path.name}"
            )
            completed += 1

    return completed

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


def find_chunk_for_video(repo_id: str, task: str, info_json: dict, modality_json: dict, episode_index: int) -> Optional[int]:
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
    decoder_module: Optional[str] = None,
    require_video: bool = True,
    save_every: int = 1,
    batch_size: int = 1,
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

        try:
            episodes_iter = iter_jsonl_from_repo(repo_id, f"{task}/meta/episodes.jsonl")
        except Exception as e:
            print(f"  ! No episodes.jsonl for {task}: {e}")
            continue

        ep_count = 0
        pending_samples: List[EpisodeSample] = []
        for rec in episodes_iter:
            try:
                episode_index = int(rec.get("episode_index"))
            except Exception:
                continue
            if episode_index < start_at_episode:
                continue

            if max_episodes_per_task is not None and ep_count >= max_episodes_per_task:
                break

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
                    episode_chunk = 0

            parquet_rel, video_rel = make_paths(info, modality, episode_index, episode_chunk)
            parquet_rel_task = f"{task}/{parquet_rel}"
            video_rel_task   = f"{task}/{video_rel}"

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
                try:
                    video_local = hf_download(repo_id, video_rel_task)
                    vr = VideoFrameReader(video_local, fps_hint=fps)
                except Exception:
                    vr = None

            table = pf.read()
            cols = {name: table[name].to_numpy() for name in table.column_names}

            sample = EpisodeSample(
                episode_index=episode_index,
                episode_chunk=episode_chunk,
                parquet_rel_task=parquet_rel_task,
                video_rel_task=video_rel_task,
                columns=cols,
                length=len(table),
                fps=fps,
                video_reader=vr,
                latent_path=lat_out,
                meta_path=meta_out,
            )

            pending_samples.append(sample)

            allowed_remaining = (
                max_episodes_per_task - ep_count if max_episodes_per_task is not None else None
            )
            should_flush = len(pending_samples) >= batch_size
            if allowed_remaining is not None and allowed_remaining <= len(pending_samples):
                should_flush = True

            if should_flush:
                to_run: Sequence[EpisodeSample]
                if allowed_remaining is not None:
                    allowed = max(0, allowed_remaining)
                    to_run = pending_samples[:allowed]
                else:
                    to_run = pending_samples

                if to_run:
                    completed = run_episode_batch(
                        to_run,
                        policy,
                        hook,
                        modality,
                        task_text_map,
                        require_video,
                        batch_size,
                    )
                    ep_count += completed
                    pending_samples = pending_samples[len(to_run) :]

        if pending_samples and (max_episodes_per_task is None or ep_count < max_episodes_per_task):
            remaining = pending_samples[:]
            if max_episodes_per_task is not None:
                remaining = remaining[: max(0, max_episodes_per_task - ep_count)]
            completed = run_episode_batch(
                remaining,
                policy,
                hook,
                modality,
                task_text_map,
                require_video,
                batch_size,
            )
            ep_count += completed
            pending_samples = []

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
    p.add_argument("--save-every", type=int, default=1, help="Flush cadence (reserved)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of episodes to process concurrently (per forward pass batch)",
    )
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
        save_every=args.save_every,
        batch_size=max(1, args.batch_size),
    )
