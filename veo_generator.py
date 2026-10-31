"""Veo3 Pro Generator — core application module.

Provides the shared orchestration layer used by both the CLI
(``python -m veo_generator``) and the Flet GUI (``veo_gui.py``).
Media processing is done with ffmpeg/ffprobe subprocesses only — no
OpenCV / NumPy / MoviePy dependencies.

Features:
    * Text/Image-to-Video generation via Veo 3.1 (with polling, retries, backoff)
    * Text-to-Image generation (Imagen / NanoBanana)
    * Music generation via Lyria realtime
    * Video chaining, batch (concurrent) processing and Bridge interpolation
    * Prompt-engineering helpers (LLM expansion, templates, history)
    * Resume state for long chains, centralized logging, type hints
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

from task_journal import TaskJournal


class CancelledError(RuntimeError):
    """Raised inside worker threads when the user requests cancellation."""

# Load environment variables from .env (config lives outside the code)
load_dotenv()

logger = logging.getLogger("veo")

# --- Configuration Constants ---


@dataclass
class AppConfig:
    """Runtime configuration, overridable via environment / .env file."""

    # Official stable Vertex AI model ID (ai.google.dev Veo 3.1 docs); the
    # ``-preview`` variant is deprecated. Override via .env if needed.
    MODEL_VIDEO: str = os.getenv("VEO_MODEL_VIDEO", "veo-3.1-generate-001")
    MODEL_IMAGE: str = os.getenv("VEO_MODEL_IMAGE", "gemini-2.5-flash-image")
    MODEL_TEXT: str = os.getenv("VEO_MODEL_TEXT", "gemini-2.5-flash")
    MODEL_AUDIO: str = os.getenv("VEO_MODEL_AUDIO", "lyria-002")
    TOKEN_LIMIT: int = int(os.getenv("VEO_TOKEN_LIMIT", "1000"))

    # Polling / retry tuning
    POLL_INTERVAL_S: float = float(os.getenv("VEO_POLL_INTERVAL", "5"))
    POLL_TIMEOUT_S: float = float(os.getenv("VEO_POLL_TIMEOUT", "900"))
    MAX_RETRIES: int = int(os.getenv("VEO_MAX_RETRIES", "3"))
    BACKOFF_BASE_S: float = float(os.getenv("VEO_BACKOFF_BASE", "2"))

    # Batch concurrency
    BATCH_WORKERS: int = int(os.getenv("VEO_BATCH_WORKERS", "3"))

    # Upscale generated videos to 1080p after download (Veo 3.1 API flag)
    UPSCALE_DEFAULT: bool = os.getenv("VEO_UPSCALE", "false").lower() in ("1", "true", "yes")

    # Generation defaults (Veo 3.1 supports durations in whole seconds 4-60)
    MIN_DURATION_S: int = int(os.getenv("VEO_MIN_DURATION", "4"))
    MAX_DURATION_S: int = int(os.getenv("VEO_MAX_DURATION", "60"))
    DEFAULT_DURATION_S: int = int(os.getenv("VEO_DURATION", "8"))
    GENERATE_NEGATIVE_PROMPT: bool = os.getenv("VEO_AUTO_NEGATIVE", "true").lower() in ("1", "true", "yes")

    # Lyria Live Music streams 48 kHz / stereo / 16-bit PCM.
    AUDIO_SAMPLE_RATE: int = int(os.getenv("VEO_AUDIO_SAMPLE_RATE", "48000"))
    AUDIO_CHANNELS: int = int(os.getenv("VEO_AUDIO_CHANNELS", "2"))
    AUDIO_SAMPLE_WIDTH: int = 2  # 16-bit PCM (fixed by the API)

    if getattr(sys, "frozen", False):
        BASE_DIR: str = os.path.dirname(sys.executable)
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    PROMPT_TEMPLATES_FILE: str = os.path.join(BASE_DIR, "prompt_templates.json")
    HISTORY_FILE: str = os.path.join(BASE_DIR, "output", "prompt_history.json")
    RESUME_FILE: str = os.path.join(BASE_DIR, "output", "resume_state.json")
    TASKS_JOURNAL_FILE: str = os.path.join(BASE_DIR, "output", "tasks.jsonl")

    @property
    def OUTPUT_DIRS(self) -> dict[str, str]:
        return {
            "video": os.path.join(self.BASE_DIR, "output", "videos"),
            "image": os.path.join(self.BASE_DIR, "output", "images"),
            "audio": os.path.join(self.BASE_DIR, "output", "audio"),
        }

    def ensure_dirs(self) -> None:
        for path in self.OUTPUT_DIRS.values():
            os.makedirs(path, exist_ok=True)


Config = AppConfig()

_DEFAULT_TEMPLATES: dict[str, str] = {
    "cinematic": "{prompt}, cinematic lighting, shallow depth of field, film grain, 24fps",
    "documentary": "{prompt}, natural lighting, handheld camera, realistic colors",
    "anime": "{prompt}, anime style, vibrant colors, dynamic motion, studio quality",
    "product_ad": "{prompt}, clean studio background, soft reflections, commercial photography style",
}


class ApiKeyError(RuntimeError):
    """Raised when the GEMINI_API_KEY is missing or obviously invalid."""


def validate_api_key() -> str:
    """Return the API key or raise :class:`ApiKeyError` early at startup.

    Performs an offline format check (Google AI Studio keys are 35+ chars
    starting with ``AIza``); a live network validation is issued lazily by
    the first API call so no extra quota is spent on startup.
    """
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise ApiKeyError(
            "GEMINI_API_KEY is not set. Create a .env file with GEMINI_API_KEY=your_key"
        )
    placeholders = ("your_", "placeholder", "xxx", "changeme", "insert")
    if any(p in key.lower() for p in placeholders) or len(key) < 20:
        raise ApiKeyError(
            "GEMINI_API_KEY looks like a placeholder or is too short. "
            "Get a real key at https://aistudio.google.com/apikey"
        )
    if key.startswith("AIza") and len(key) < 35:
        raise ApiKeyError(
            f"GEMINI_API_KEY has an invalid length ({len(key)}); "
            "Google API keys are 35+ characters. Check for truncation."
        )
    return key


# --- File Utilities ---


class FileManager:
    """Filename allocation, prompt history and resume-state persistence."""

    _history_lock = threading.Lock()
    _resume_lock = threading.Lock()

    @staticmethod
    def get_unique_filename(folder: str, base_name: str, extension: str, strict: bool = False) -> str:
        """Generate a unique filename in ``folder``.

        With ``strict=True`` prompts on stdin for a new name on collision,
        otherwise appends an incrementing counter (``video_1.mp4``).
        """
        if not base_name:
            base_name = "generated"

        if not extension.startswith("."):
            extension = "." + extension

        if base_name.lower().endswith(extension):
            base_name = base_name[: -len(extension)]

        full_path = os.path.join(folder, base_name + extension)

        if strict:
            while os.path.exists(full_path):
                logger.error("File '%s%s' already exists in %s.", base_name, extension, folder)
                base_name = input("Enter a new name (without extension): ").strip()
                full_path = os.path.join(folder, base_name + extension)
            return full_path

        counter = 1
        new_base_name = base_name
        while os.path.exists(full_path):
            new_base_name = f"{base_name}_{counter}"
            full_path = os.path.join(folder, new_base_name + extension)
            counter += 1

        return full_path

    @staticmethod
    def ask_for_filename(prompt_text: str, default_name: str, folder: str, extension: str) -> str:
        name = input(prompt_text)
        if not name.strip():
            name = default_name
        return FileManager.get_unique_filename(folder, name, extension, strict=True)

    @staticmethod
    def cleanup_files(file_list: Iterable[Optional[str]]) -> None:
        logger.info("Cleaning up temporary files...")
        for f in file_list:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError as e:
                    logger.warning("Could not remove %s: %s", f, e)

    # ---- Prompt history (item 14) ----

    @staticmethod
    def record_prompt(prompt: str, kind: str = "video", output: Optional[str] = None) -> None:
        """Append a prompt to the persistent history file (thread-safe)."""
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "prompt": prompt, "output": output}
        try:
            with FileManager._history_lock:
                os.makedirs(os.path.dirname(Config.HISTORY_FILE), exist_ok=True)
                history = FileManager.load_history(limit=0)
                history.append(entry)
                with open(Config.HISTORY_FILE, "w", encoding="utf-8") as fh:
                    json.dump(history[-200:], fh, ensure_ascii=False, indent=1)
        except OSError as e:
            logger.warning("Failed to write prompt history: %s", e)

    @staticmethod
    def load_history(limit: int = 10) -> list[dict]:
        try:
            with open(Config.HISTORY_FILE, "r", encoding="utf-8") as fh:
                history = json.load(fh)
            return history[-limit:] if limit else history
        except (OSError, json.JSONDecodeError):
            return []

    @staticmethod
    def load_templates() -> dict[str, str]:
        """Load user templates from JSON, falling back to built-ins."""
        try:
            with open(Config.PROMPT_TEMPLATES_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and data:
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return dict(_DEFAULT_TEMPLATES)

    @staticmethod
    def apply_template(template_name: str, prompt: str) -> str:
        templates = FileManager.load_templates()
        tpl = templates.get(template_name)
        return tpl.format(prompt=prompt) if tpl else prompt

    # ---- Resume state (item 15) ----

    @staticmethod
    def save_resume_state(state: dict) -> None:
        """Atomically persist chain resume state under a lock.

        Written to a temp file in the same directory and moved over the
        target so a crash mid-write can never corrupt the resume file,
        even when batch worker threads run concurrently.
        """
        try:
            os.makedirs(os.path.dirname(Config.RESUME_FILE), exist_ok=True)
            with FileManager._resume_lock:
                fd, tmp_path = tempfile.mkstemp(
                    dir=os.path.dirname(Config.RESUME_FILE), prefix=".resume_", suffix=".tmp"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        json.dump(state, fh, ensure_ascii=False, indent=1)
                    os.replace(tmp_path, Config.RESUME_FILE)
                except BaseException:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    raise
        except OSError as e:
            logger.warning("Failed to save resume state: %s", e)

    @staticmethod
    def load_resume_state() -> Optional[dict]:
        try:
            with open(Config.RESUME_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def clear_resume_state() -> None:
        try:
            if os.path.exists(Config.RESUME_FILE):
                os.remove(Config.RESUME_FILE)
        except OSError as e:
            logger.warning("Failed to clear resume state: %s", e)


# --- Media Processing (ffmpeg / ffprobe) ---


class MediaProcessor:
    """Frame extraction, normalization and concatenation via ffmpeg.

    All heavy video work is delegated to the ``ffmpeg``/``ffprobe``
    binaries (subprocess), which keeps the dependency footprint small —
    no OpenCV / NumPy / MoviePy required.
    """

    @staticmethod
    def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
        """Run an ffmpeg/ffprobe command quietly; raise on failure."""
        logger.debug("Running: %s", " ".join(cmd))
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )

    @staticmethod
    def extract_frame(video_path: str, output_image_path: str, pos: str = "last") -> Optional[str]:
        """Extract the first or last frame of a video using ffmpeg.

        ``-sseof -0.1`` seeks near the true end of the stream (robust for
        VBR/B-frame files where a reported frame count can be wrong); if
        nothing is decoded there we fall back to scanning from the start
        and keeping the final decoded frame.
        """
        logger.info("Extracting %s frame from %s...", pos, video_path)
        try:
            if pos == "last":
                # Fast path: seek 0.1s before the end, grab one frame.
                try:
                    MediaProcessor._run([
                        "ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path,
                        "-map", "v:0", "-frames:v", "1", "-update", "1",
                        "-q:v", "2", output_image_path,
                    ], timeout=120)
                    if os.path.exists(output_image_path) and os.path.getsize(output_image_path) > 0:
                        return output_image_path
                except subprocess.CalledProcessError:
                    pass
                # Fallback: decode whole stream, keep the final frame.
                MediaProcessor._run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-map", "v:0", "-vsync", "vfr", "-f", "null", os.devnull,
                ])
                tmpdir = tempfile.mkdtemp(prefix="veo_frames_")
                try:
                    MediaProcessor._run([
                        "ffmpeg", "-y", "-i", video_path, "-map", "v:0",
                        os.path.join(tmpdir, "f%08d.png"),
                    ])
                    frames = sorted(f for f in os.listdir(tmpdir) if f.endswith(".png"))
                    if not frames:
                        logger.error("No frames decoded from %s", video_path)
                        return None
                    shutil.copyfile(os.path.join(tmpdir, frames[-1]), output_image_path)
                    return output_image_path
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
            ret = MediaProcessor._run([
                "ffmpeg", "-y", "-i", video_path,
                "-map", "v:0", "-frames:v", "1", "-update", "1",
                "-q:v", "2", output_image_path,
            ], timeout=120)
            if os.path.exists(output_image_path) and os.path.getsize(output_image_path) > 0:
                return output_image_path
            logger.error("Failed to read frame from %s (%s)", video_path, ret.stderr[-300:])
            return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            logger.error("Frame extraction failed for %s: %s", video_path, e)
            return None

    @staticmethod
    def extract_thumbnail(video_path: str, output_image_path: str, width: int = 320) -> Optional[str]:
        """Save a small JPEG thumbnail (first frame) for GUI previews."""
        try:
            MediaProcessor._run([
                "ffmpeg", "-y", "-i", video_path,
                "-map", "v:0", "-frames:v", "1",
                "-vf", f"scale={width}:-2", "-q:v", "5", output_image_path,
            ], timeout=60)
            if os.path.exists(output_image_path) and os.path.getsize(output_image_path) > 0:
                return output_image_path
            return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            logger.error("Thumbnail extraction failed for %s: %s", video_path, e)
            return None

    @staticmethod
    def _probe_video(path: str) -> Optional[dict]:
        """Return {w, h, fps} metadata of a video file via ffprobe, or None."""
        try:
            proc = MediaProcessor._run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json", path,
            ], timeout=60)
            data = json.loads(proc.stdout or "{}")
            streams = data.get("streams") or []
            if not streams:
                return None
            s = streams[0]
            w, h = int(s.get("width") or 0), int(s.get("height") or 0)
            rate = s.get("r_frame_rate") or "24/1"
            num, _, den = rate.partition("/")
            try:
                fps = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                fps = 24.0
            if w <= 0 or h <= 0:
                return None
            return {"w": w, "h": h, "fps": fps if fps > 0 else 24.0}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                OSError, ValueError, json.JSONDecodeError) as e:
            logger.error("ffprobe failed for %s: %s", path, e)
            return None

    @staticmethod
    def normalize_clip(path: str, target_w: int, target_h: int, target_fps: float) -> Optional[str]:
        """Re-encode ``path`` to the target size/fps (letterboxed, no cropping).

        Returns the normalized file path, or the original path when it
        already matches the target, or ``None`` on failure.
        """
        meta = MediaProcessor._probe_video(path)
        if meta is None:
            return None
        if meta["w"] == target_w and meta["h"] == target_h and abs(meta["fps"] - target_fps) < 0.5:
            return path

        out_path = os.path.join(
            os.path.dirname(path), "_norm_" + os.path.basename(path)
        )
        # Preserve aspect ratio: scale to fit inside the target canvas,
        # then pad with black (force even dimensions for libx264).
        vf = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={target_fps:.6f}"
        )
        try:
            MediaProcessor._run([
                "ffmpeg", "-y", "-i", path,
                "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-ar", "48000", "-ac", "2",
                out_path,
            ])
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise RuntimeError("ffmpeg produced no output")
            logger.info("Normalized %s -> %dx%d@%.2ffps", os.path.basename(path), target_w, target_h, target_fps)
            return out_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as e:
            logger.error("Normalization failed for %s: %s", path, e)
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            return None

    @staticmethod
    def concatenate_videos(video_paths: list[str], output_filename: str) -> Optional[str]:
        """Concatenate video files with automatic normalization.

        Veo can return clips with differing resolution/FPS (e.g. after an
        upscale or across model revisions); concatenating such streams
        produces desynchronized output. All clips are therefore re-encoded
        to the geometry of the first clip, then stitched with the ffmpeg
        concat demuxer (bit-exact join of identically-encoded streams).
        """
        logger.info("Concatenating %d videos...", len(video_paths))
        normalized_files: list[str] = []
        list_file: Optional[str] = None
        try:
            metas = [MediaProcessor._probe_video(p) for p in video_paths]
            if any(m is None for m in metas):
                raise RuntimeError("Could not probe one of the input videos")
            assert metas and metas[0] is not None
            tw, th = metas[0]["w"], metas[0]["h"]
            tfps = sum(m["fps"] for m in metas) / len(metas)  # type: ignore[index]

            use_paths: list[str] = []
            for p, m in zip(video_paths, metas):
                assert m is not None
                if m["w"] == tw and m["h"] == th and abs(m["fps"] - tfps) < 0.5:
                    use_paths.append(p)
                    continue
                norm = MediaProcessor.normalize_clip(p, tw, th, tfps)
                if norm is None:
                    raise RuntimeError(f"Normalization failed for {p}")
                if norm != p:
                    normalized_files.append(norm)
                use_paths.append(norm)

            # Write a concat-list file (single-quoted paths escaped).
            fd, list_file = tempfile.mkstemp(prefix="veo_concat_", suffix=".txt",
                                             dir=os.path.dirname(os.path.abspath(output_filename)))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for up in use_paths:
                    safe = os.path.abspath(up).replace("'", "'\\''")
                    fh.write(f"file '{safe}'\n")

            MediaProcessor._run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", output_filename,
            ])
            if not os.path.exists(output_filename) or os.path.getsize(output_filename) == 0:
                raise RuntimeError("ffmpeg produced no concatenated output")
            logger.info("Combined video saved: %s", output_filename)
            return output_filename
        except Exception as e:  # noqa: BLE001 - report and degrade gracefully
            logger.error("Video concatenation failed: %s", e)
            return None
        finally:
            for nf in normalized_files:
                try:
                    if os.path.exists(nf):
                        os.remove(nf)
                except OSError:
                    pass
            if list_file is not None:
                try:
                    if os.path.exists(list_file):
                        os.remove(list_file)
                except OSError:
                    pass


# --- Generative AI Service ---


class GenAIService:
    """Thin, resilient wrapper around the Google GenAI client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or validate_api_key()
        self.client = genai.Client(api_key=self.api_key)
        # Cooperative cancellation: set the Event to abort in-flight work.
        self.cancel_event: threading.Event = threading.Event()
        # JSONL task journal (statistics / observability).
        self._journal = TaskJournal(Config.TASKS_JOURNAL_FILE)

    def cancel(self) -> None:
        """Request cancellation of the current generation/polling loop."""
        self.cancel_event.set()

    def reset_cancel(self) -> None:
        self.cancel_event.clear()

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError("Cancelled by user")

    # ---- low-level helpers ----

    @staticmethod
    def _is_retryable(err: Exception) -> bool:
        err_str = str(err).lower()
        return any(tok in err_str for tok in ("429", "rate limit", "resource exhausted", "500", "502", "503", "deadline", "unavailable", "timeout"))

    def _retry(self, fn: Callable[[], object], description: str) -> object:
        """Run ``fn`` with exponential backoff + jitter on transient errors."""
        attempt = 0
        while True:
            try:
                return fn()
            except CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._check_cancelled()
                attempt += 1
                if attempt >= Config.MAX_RETRIES or not self._is_retryable(e):
                    raise
                delay = min(60.0, Config.BACKOFF_BASE_S * (2 ** (attempt - 1))) + random.uniform(0, 1)
                logger.warning("%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                               description, attempt, Config.MAX_RETRIES, e, delay)
                time.sleep(delay)

    def _await_operation(self, operation, progress_cb: Optional[Callable[[Optional[int]], None]] = None) -> None:
        """Poll an LRO with timeout; raise on timeout, transport errors or cancel.

        If the API exposes ``metadata.progress`` it is forwarded to
        ``progress_cb`` so frontends can render a real progress bar.
        """
        start = time.time()
        last_pct: Optional[int] = None
        while not getattr(operation, "done", False):
            self._check_cancelled()
            if time.time() - start > Config.POLL_TIMEOUT_S:
                raise TimeoutError(f"Generation exceeded {Config.POLL_TIMEOUT_S:.0f}s timeout.")
            meta = getattr(operation, "metadata", None)
            pct = getattr(meta, "progress", None) if meta is not None else None
            if isinstance(pct, int) and pct != last_pct:
                last_pct = pct
                if progress_cb:
                    progress_cb(pct)
            time.sleep(Config.POLL_INTERVAL_S)
            operation = self.client.operations.get(operation)
        if progress_cb:
            progress_cb(100)

    def _get_image_param(self, image_path: str) -> Optional[types.Image]:
        """Read an image file and return a ``types.Image`` object."""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            mime_type = "image/png"
            if image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            return types.Image.model_construct(imageBytes=image_bytes, mimeType=mime_type)
        except OSError as e:
            logger.error("Failed to read image %s: %s", image_path, e)
            return None

    # ---- text helpers ----

    def count_tokens(self, text: str) -> int:
        try:
            resp = self.client.models.count_tokens(model=Config.MODEL_TEXT, contents=text)
            return int(resp.total_tokens)
        except Exception as e:  # noqa: BLE001
            logger.debug("Token counting unavailable: %s", e)
            return 0

    def generate_negative_prompt(self, video_prompt: str) -> Optional[str]:
        try:
            prompt = (
                "Generate a concise comma-separated list of negative prompt keywords "
                "for a high quality video generation based on this positive prompt: "
                f"'{video_prompt}'. Focus on avoiding bad quality, deformations, blurs, "
                "watermark, text, ugly, distorted. Return ONLY the keywords."
            )
            resp = self.client.models.generate_content(model=Config.MODEL_TEXT, contents=prompt)
            return resp.text.strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("Negative prompt generation failed: %s", e)
            return None

    def expand_prompt(self, short_prompt: str) -> Optional[str]:
        """Prompt-engineering helper: expand a terse idea into a rich Veo prompt."""
        try:
            meta = (
                "You are an expert prompt engineer for Google Veo video generation. "
                "Rewrite the following idea into one detailed English prompt (max 120 words) "
                "covering subject, action, camera movement, lighting, mood and style. "
                "Return ONLY the prompt text.\n\nIdea: " + short_prompt
            )
            resp = self.client.models.generate_content(model=Config.MODEL_TEXT, contents=meta)
            return resp.text.strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("Prompt expansion failed: %s", e)
            return None

    # ---- generation ----

    @staticmethod
    def parse_reference_spec(spec: str) -> tuple[str, str]:
        """Split a reference spec into (path, reference_type).

        Explicit prefixes are preferred over filename heuristics:
            ``style:/path/img.png`` or ``asset:/path/img.png``  -> prefix wins
            ``/path/style_img.png``                             -> legacy heuristic (STYLE)
            ``/path/img.png``                                   -> ASSET (default)
        """
        low = spec.lower()
        for prefix, kind in (("style:", "STYLE"), ("asset:", "ASSET")):
            if low.startswith(prefix):
                return spec.split(":", 1)[1].strip(), kind
        # Legacy heuristic kept for backwards compatibility.
        if "style" in os.path.basename(spec).lower():
            return spec, "STYLE"
        return spec, "ASSET"

    def generate_video(
        self,
        prompt: str,
        output_filename: str,
        image_path: Optional[str] = None,
        last_frame_path: Optional[str] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: Optional[int] = None,
        generate_audio: bool = True,
        negative_prompt: Optional[str] = None,
        reference_image_paths: Optional[list[str]] = None,
        upsampling: Optional[bool] = None,
        progress_cb: Optional[Callable[[Optional[int]], None]] = None,
    ) -> Optional[str]:
        """Generate a single Veo clip. Returns the output path or ``None``.

        Parameter constraints follow the official Veo 3.1 docs:
        * resolution: 720p / 1080p (4k only for veo-3.1-generate-preview);
        * duration_seconds: whole seconds in [4, 60];
        * reference images: ASSET up to 3, STYLE up to 3 per request;
        * upsampling: Veo 3.1 native 1080p/4k upscale pass (paid tier).
        """
        logger.info("Generating video: '%s' [%s, %s]", prompt, resolution, aspect_ratio)
        FileManager.record_prompt(prompt, "video", output_filename)
        self._check_cancelled()
        start = time.time()

        tokens = self.count_tokens(prompt)
        if tokens > Config.TOKEN_LIMIT:
            logger.warning("Prompt is very long (%d tokens). Limit: %d.", tokens, Config.TOKEN_LIMIT)

        try:
            config_params: dict = {
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "person_generation": "allow_adult",
                "generate_audio": generate_audio,
            }
            use_upscale = Config.UPSCALE_DEFAULT if upsampling is None else upsampling
            if use_upscale:
                config_params["upsampling"] = True
            if duration_seconds:
                dur = int(duration_seconds)
                if not Config.MIN_DURATION_S <= dur <= Config.MAX_DURATION_S:
                    dur = max(Config.MIN_DURATION_S, min(dur, Config.MAX_DURATION_S))
                    logger.warning(
                        "Duration must be %d-%d seconds; adjusted to %d.",
                        Config.MIN_DURATION_S, Config.MAX_DURATION_S, dur,
                    )
                config_params["duration_seconds"] = dur

            if negative_prompt:
                config_params["negative_prompt"] = negative_prompt

            if reference_image_paths:
                refs = []
                # Official limits: max 3 ASSET and max 3 STYLE references.
                asset_left, style_left = 3, 3
                for spec in reference_image_paths:
                    path, kind = self.parse_reference_spec(spec)
                    if kind == "STYLE":
                        if style_left == 0:
                            logger.warning("Ignoring extra STYLE reference (max 3): %s", path)
                            continue
                        style_left -= 1
                    else:
                        if asset_left == 0:
                            logger.warning("Ignoring extra ASSET reference (max 3): %s", path)
                            continue
                        asset_left -= 1
                    img_obj = self._get_image_param(path)
                    if img_obj:
                        refs.append(types.VideoGenerationReferenceImage(image=img_obj, reference_type=kind))
                if refs:
                    config_params["reference_images"] = refs

            image_param = None
            if image_path:
                image_param = self._get_image_param(image_path)
                if not image_param:
                    return None

            if last_frame_path:
                last_frame_param = self._get_image_param(last_frame_path)
                if last_frame_param:
                    config_params["last_frame"] = last_frame_param

            def submit(cfg_params: dict):
                cfg = types.GenerateVideosConfig(**cfg_params)
                return self.client.models.generate_videos(
                    model=Config.MODEL_VIDEO,
                    prompt=prompt,
                    image=image_param,
                    config=cfg,
                )

            # Try with person_generation first; fall back without it if rejected.
            try:
                operation = self._retry(lambda: submit(config_params), "Video request")
            except Exception as e:  # noqa: BLE001
                err_str = str(e)
                if "INVALID_ARGUMENT" in err_str and ("personGeneration" in err_str or "allow_adult" in err_str):
                    logger.warning("person_generation argument rejected; retrying without it...")
                    config_params.pop("person_generation", None)
                    operation = self._retry(lambda: submit(config_params), "Video request (fallback)")
                else:
                    raise

            logger.info("Waiting for generation to finish...")
            self._await_operation(operation, progress_cb=progress_cb)

            generated_video = operation.response.generated_videos[0]
            self.client.files.download(file=generated_video.video)
            generated_video.video.save(output_filename)
            logger.info("Video saved: %s", output_filename)
            self._journal.record("video", "ok", prompt=prompt, output=output_filename,
                                 duration_s=time.time() - start,
                                 extra={"resolution": resolution, "upscaled": bool(use_upscale)})
            return output_filename
        except CancelledError:
            logger.warning("Video generation cancelled by user.")
            self._journal.record("video", "failed", prompt=prompt, error="cancelled",
                                 duration_s=time.time() - start)
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("Video generation failed: %s", e)
            self._journal.record("video", "failed", prompt=prompt, error=str(e),
                                 duration_s=time.time() - start)
            return None

    def generate_image(self, prompt: str, output_filename: str) -> Optional[str]:
        logger.info("Generating image: '%s'", prompt)
        FileManager.record_prompt(prompt, "image", output_filename)
        self._check_cancelled()
        start = time.time()
        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_IMAGE,
                contents=[prompt],
                config={"response_modalities": ["IMAGE"]},
            )
            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    image.save(output_filename)
                    logger.info("Image saved: %s", output_filename)
                    self._journal.record("image", "ok", prompt=prompt, output=output_filename,
                                         duration_s=time.time() - start)
                    return output_filename
            logger.error("No image returned in response parts.")
            self._journal.record("image", "failed", prompt=prompt, error="No image in response")
            return None
        except CancelledError:
            self._journal.record("image", "failed", prompt=prompt, error="cancelled")
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("Image generation failed: %s", e)
            self._journal.record("image", "failed", prompt=prompt, error=str(e))
            return None

    def generate_audio(
        self,
        prompt: str,
        duration_seconds: int = 10,
        output_filename: Optional[str] = None,
        bpm: int = 120,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> Optional[str]:
        """Stream music from Lyria (Live Music API, v1alpha) into a WAV file.

        Note: per official docs the Live Music API is available on paid
        tiers only; free-tier keys receive a permission error here (Veo
        itself remains usable via ``generate_video``).
        """
        if not output_filename:
            output_filename = FileManager.get_unique_filename(Config.OUTPUT_DIRS["audio"], "generated_music", ".wav")

        logger.info("Generating music: '%s' (%d sec)...", prompt, duration_seconds)
        FileManager.record_prompt(prompt, "audio", output_filename)
        self._check_cancelled()
        start = time.time()

        async def run_audio_session():
            client_lyria = genai.Client(
                api_key=self.api_key,
                http_options={"api_version": "v1alpha"},
            )
            audio_buffer = bytearray()
            channels, sample_width, framerate = (
                Config.AUDIO_CHANNELS,
                Config.AUDIO_SAMPLE_WIDTH,
                Config.AUDIO_SAMPLE_RATE,
            )
            try:
                async with client_lyria.aio.live.music.connect(model=Config.MODEL_AUDIO) as session:
                    await session.set_weighted_prompts(prompts=[types.WeightedPrompt(text=prompt, weight=1.0)])
                    cfg_params: dict = {"bpm": bpm, "temperature": temperature}
                    if seed is not None:
                        cfg_params["seed"] = seed
                    await session.set_music_generation_config(
                        config=types.LiveMusicGenerationConfig(**cfg_params)
                    )
                    await session.play()
                    logger.info("Recording audio stream...")
                    start_time = time.time()
                    async for message in session.receive():
                        if self.cancel_event.is_set():
                            raise CancelledError("Cancelled by user")
                        sc = getattr(message, "server_content", None)
                        setup = getattr(sc, "setup", None) if sc else None
                        if setup is not None:
                            # Read actual audio params from the API handshake
                            # instead of hard-coding them (item 16).
                            framerate = getattr(setup, "audio_sample_rate", None) or framerate
                            channels = getattr(setup, "input_audio_channels", None) or channels
                        if sc and getattr(sc, "audio_chunks", None):
                            for chunk in sc.audio_chunks:
                                audio_buffer.extend(chunk.data)
                        if time.time() - start_time > duration_seconds:
                            break
                    logger.info("Audio generation finished.")
                    return bytes(audio_buffer), framerate, channels, sample_width
            except CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                err = str(e)
                if "PERMISSION_DENIED" in err or "403" in err:
                    logger.error(
                        "Lyria access denied — the Live Music API requires a "
                        "paid (Tier 1+) API key. Veo video generation is unaffected."
                    )
                else:
                    logger.error("Lyria session error: %s", e)
                return None

        try:
            result = asyncio.run(run_audio_session())
            if result:
                audio_data, framerate, channels, sample_width = result
                with wave.open(output_filename, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(framerate)
                    wf.writeframes(audio_data)
                logger.info("Audio saved: %s", output_filename)
                self._journal.record("audio", "ok", prompt=prompt, output=output_filename,
                                     duration_s=time.time() - start)
                return output_filename
            self._journal.record("audio", "failed", prompt=prompt, error="session returned no audio",
                                 duration_s=time.time() - start)
            return None
        except CancelledError:
            self._journal.record("audio", "failed", prompt=prompt, error="cancelled",
                                 duration_s=time.time() - start)
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("Audio generation launch failed: %s", e)
            self._journal.record("audio", "failed", prompt=prompt, error=str(e),
                                 duration_s=time.time() - start)
            return None


# --- Orchestrator: shared high-level flows for CLI and GUI ---


class GenerationOrchestrator:
    """High-level workflows (single, chain, batch, bridge) reused by frontends.

    All progress reporting goes through the injected ``log_fn`` callback so the
    CLI can print to stdout while the GUI routes messages to its console widget
    without monkey-patching ``sys.stdout`` (item 2).

    ``progress_cb(percent|None, stage)`` is an optional second channel used by
    GUI progress bars; it never raises into the workflow.
    """

    def __init__(
        self,
        ai: Optional[GenAIService] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[Optional[int], str], None]] = None,
    ):
        self.ai = ai or GenAIService()
        self.log = log_fn or (lambda msg: print(msg, flush=True))
        self._progress_cb = progress_cb

    def _report_progress(self, pct: Optional[int], stage: str = "") -> None:
        if self._progress_cb is not None:
            try:
                self._progress_cb(pct, stage)
            except Exception:  # noqa: BLE001 - UI callbacks must not break work
                pass

    def cancel(self) -> None:
        """Request cancellation of any in-flight orchestrator workflow."""
        self.ai.cancel()

    # ---------- single ----------

    def single_video(
        self,
        prompt: str,
        output_filename: Optional[str] = None,
        *,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration_seconds: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        auto_negative: Optional[bool] = None,
        template: Optional[str] = None,
        expand: bool = False,
        image_path: Optional[str] = None,
        last_frame_path: Optional[str] = None,
        reference_image_paths: Optional[list[str]] = None,
        generate_audio: bool = True,
        upsampling: Optional[bool] = None,
    ) -> Optional[str]:
        self.ai.reset_cancel()
        start = time.time()
        status = "failed"
        out: Optional[str] = None
        try:
            if expand:
                self._report_progress(0, "expanding prompt")
                expanded = self.ai.expand_prompt(prompt)
                if expanded:
                    self.log(f"Expanded prompt: {expanded}")
                    prompt = expanded
            if template:
                prompt = FileManager.apply_template(template, prompt)

            use_auto = Config.GENERATE_NEGATIVE_PROMPT if auto_negative is None else auto_negative
            if not negative_prompt and use_auto:
                self._report_progress(0, "negative prompt")
                self.log("Auto-generating negative prompt...")
                negative_prompt = self.ai.generate_negative_prompt(prompt)
                if negative_prompt:
                    self.log(f"Negative prompt: {negative_prompt}")

            if output_filename is None:
                output_filename = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "veo_video", ".mp4")

            self._report_progress(5, "submitting")
            out = self.ai.generate_video(
                prompt,
                output_filename,
                image_path=image_path,
                last_frame_path=last_frame_path,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                duration_seconds=duration_seconds,
                generate_audio=generate_audio,
                negative_prompt=negative_prompt,
                reference_image_paths=reference_image_paths,
                upsampling=upsampling,
                progress_cb=lambda pct: self._report_progress(
                    (10 + pct * 88 // 100) if pct is not None else None, "generating"
                ),
            )
            status = "ok" if out else "failed"
            return out
        finally:
            self._report_progress(100 if out else None, "done" if out else "stopped")
            self.ai._journal.record("task_video", status, prompt=prompt, output=out,
                                    duration_s=time.time() - start)

    # ---------- batch (concurrent, item 10) ----------

    def batch_generate(
        self,
        prompts: list[str],
        resolution: str = "720p",
        max_workers: Optional[int] = None,
        generate_audio: bool = True,
        upsampling: Optional[bool] = None,
    ) -> list[Optional[str]]:
        self.ai.reset_cancel()
        start = time.time()
        results: list[Optional[str]] = [None] * len(prompts)
        workers = max_workers or Config.BATCH_WORKERS
        done_count = {"n": 0}

        def job(idx_prompt: tuple[int, str]) -> None:
            idx, p = idx_prompt
            if self.ai.cancel_event.is_set():
                self.log(f"Batch item {idx + 1} skipped (cancelled).")
                return
            fname = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], f"batch_{idx + 1}", ".mp4")
            try:
                results[idx] = self.ai.generate_video(
                    p, fname, resolution=resolution,
                    generate_audio=generate_audio, upsampling=upsampling,
                )
            except CancelledError:
                self.log(f"Batch item {idx + 1} cancelled.")
            except Exception as e:  # noqa: BLE001
                self.log(f"Batch item {idx + 1} failed: {e}")
            finally:
                done_count["n"] += 1
                self._report_progress(100 * done_count["n"] // len(prompts), f"batch {done_count['n']}/{len(prompts)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(job, list(enumerate(prompts))))

        ok = sum(1 for r in results if r)
        self.log(f"Batch finished: {ok}/{len(prompts)} succeeded.")
        self.ai._journal.record("batch", "ok" if ok == len(prompts) else "failed",
                                prompt=f"{len(prompts)} prompts", duration_s=time.time() - start,
                                extra={"succeeded": ok, "total": len(prompts)})
        return results

    # ---------- chain (with resume, item 15) ----------

    def run_chain(
        self,
        prompts: list[str],
        *,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        start_image: Optional[str] = None,
        final_name: Optional[str] = None,
        generate_audio: bool = True,
        upsampling: Optional[bool] = None,
    ) -> Optional[str]:
        """Generate a chain where each segment starts from the previous last frame.

        State is persisted after every successful link so an interrupted chain
        can be resumed with :meth:`resume_chain`.  Works for N clips (any
        number of prompts); cancellation stops cleanly and keeps resume state.
        """
        if not prompts:
            self.log("Chain requires at least one prompt.")
            return None

        self.ai.reset_cancel()
        start = time.time()
        video_chain: list[str] = []
        temp_files: list[str] = []
        remaining = list(prompts)
        total = len(prompts)

        first_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "chain_segment_1", ".mp4")
        kwargs = dict(aspect_ratio=aspect_ratio, resolution=resolution, generate_audio=generate_audio,
                      upsampling=upsampling)
        self._report_progress(5, "segment 1/{}".format(total))
        v = self.ai.generate_video(remaining.pop(0), first_name, image_path=start_image, **kwargs)
        if not v:
            self.log("First chain segment failed.")
            self.ai._journal.record("chain", "failed", prompt=f"1/{total}", duration_s=time.time() - start)
            return None
        video_chain.append(v)
        temp_files.append(v)
        self._report_progress(int(100 * len(video_chain) / total), f"segment {len(video_chain)}/{total}")
        FileManager.save_resume_state({"prompts": remaining, "chain": video_chain, "temp_files": temp_files,
                                       "aspect_ratio": aspect_ratio, "resolution": resolution})

        cancelled = False
        for i, p in enumerate(remaining):
            if self.ai.cancel_event.is_set():
                cancelled = True
                self.log("Chain cancelled; state saved for resume.")
                remaining = remaining[i:]
                FileManager.save_resume_state({"prompts": remaining, "chain": video_chain,
                                               "temp_files": temp_files, **kwargs})
                break
            last_frame_path = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], f"temp_frame_{i}", ".png")
            if not MediaProcessor.extract_frame(video_chain[-1], last_frame_path, "last"):
                self.log("Frame extraction failed; stopping chain (state saved for resume).")
                remaining = remaining[i:]
                FileManager.save_resume_state({"prompts": remaining, "chain": video_chain,
                                               "temp_files": temp_files, **kwargs})
                break
            temp_files.append(last_frame_path)
            next_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], f"chain_segment_{i + 2}", ".mp4")
            try:
                v_next = self.ai.generate_video(p or "Continue the action, high quality", next_name,
                                                image_path=last_frame_path, **kwargs)
            except CancelledError:
                cancelled = True
                v_next = None
            if not v_next:
                self.log(f"Segment {i + 2} failed; partial chain kept (state saved for resume).")
                remaining = remaining[i:]
                FileManager.save_resume_state({"prompts": remaining, "chain": video_chain,
                                               "temp_files": temp_files, **kwargs})
                break
            video_chain.append(v_next)
            temp_files.append(v_next)
            self._report_progress(int(100 * len(video_chain) / total), f"segment {len(video_chain)}/{total}")
            FileManager.save_resume_state({"prompts": [], "chain": video_chain, "temp_files": temp_files, **kwargs})

        stitched = None if cancelled else self._finalize_chain(video_chain, temp_files, final_name)
        if stitched:
            FileManager.clear_resume_state()
        self.ai._journal.record("chain", "ok" if stitched else "failed",
                                prompt=f"{total} segments", output=stitched,
                                duration_s=time.time() - start,
                                extra={"segments_done": len(video_chain)})
        return stitched

    def resume_chain(self, final_name: Optional[str] = None) -> Optional[str]:
        state = FileManager.load_resume_state()
        if not state:
            self.log("No saved chain state found.")
            return None
        pending = state.get("prompts") or []
        if pending:
            self.log(f"Resuming chain: {len(pending)} segments remaining.")
            # Continue from the last successfully generated segment's frame.
            last_seg = (state.get("chain") or [None])[-1]
            start_image = None
            if last_seg and os.path.exists(last_seg):
                start_image = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "resume_frame", ".png")
                if not MediaProcessor.extract_frame(last_seg, start_image, "last"):
                    start_image = None
            return self.run_chain(
                pending,
                aspect_ratio=state.get("aspect_ratio", "16:9"),
                resolution=state.get("resolution", "720p"),
                start_image=start_image,
                final_name=final_name,
            )
        # Nothing pending: just stitch what we have.
        return self._finalize_chain(state.get("chain", []), state.get("temp_files", []), final_name)

    def _finalize_chain(self, video_chain: list[str], temp_files: list[str], final_name: Optional[str]) -> Optional[str]:
        if not video_chain:
            self.log("Chain is empty.")
            return None
        if final_name is None:
            final_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "full_chain", ".mp4")
        if len(video_chain) > 1:
            out = MediaProcessor.concatenate_videos(video_chain, final_name)
        else:
            try:
                shutil.copy(video_chain[0], final_name)
                self.log(f"Video saved: {final_name}")
                out = final_name
            except OSError as e:
                self.log(f"Copy failed: {e}")
                out = None
        FileManager.cleanup_files(temp_files)
        return out

    # ---------- bridge (shared by CLI & GUI, items 1, 8) ----------

    def bridge_step1_prepare(
        self,
        v1_path: str,
        v2_path_or_none: Optional[str],
        v2_prompt: Optional[str],
        bridge_prompt: Optional[str],
        *,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        generate_audio: bool = True,
        upsampling: Optional[bool] = None,
    ) -> Optional[dict]:
        """Extract frames and generate the bridge clip between two videos.

        Returns a bridge-state dict for :meth:`bridge_stitch_preview` /
        :meth:`bridge_regen_v2`, or ``None`` on failure.
        """
        if not v1_path or not os.path.exists(v1_path):
            self.log("Error: Video 1 not found.")
            return None

        self.ai.reset_cancel()
        state: dict = {
            "p1": v1_path,
            "bridge": None,
            "v2_type": None,
            "v2_path": None,
            "preview": None,
            "temp_files": [],
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "generate_audio": generate_audio,
            "upsampling": upsampling,
        }

        if v2_path_or_none and os.path.exists(v2_path_or_none):
            state["v2_type"] = "existing"
            state["v2_path"] = v2_path_or_none
        else:
            if not v2_prompt:
                self.log("Error: provide Video 2 path OR prompt.")
                return None
            self.log("Generating initial Video 2...")
            self._report_progress(10, "video 2")
            v2_file = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "bridge_v2", ".mp4")
            v = self.ai.generate_video(v2_prompt, v2_file, aspect_ratio=aspect_ratio, resolution=resolution,
                                       generate_audio=generate_audio, upsampling=upsampling)
            if not v:
                return None
            state["v2_type"] = "generated"
            state["v2_path"] = v
            state["temp_files"].append(v)

        f1 = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "bridge_start", ".png")
        f2 = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "bridge_end", ".png")
        state["temp_files"].extend([f1, f2])

        if not (MediaProcessor.extract_frame(state["p1"], f1, "last")
                and MediaProcessor.extract_frame(state["v2_path"], f2, "first")):
            self.log("Frame extraction failed.")
            FileManager.cleanup_files(state["temp_files"])
            return None

        b_prompt = bridge_prompt or "Smooth transition between the two scenes, cinematic"
        b_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "bridge_seg", ".mp4")
        if not self.ai.generate_video(b_prompt, b_name, image_path=f1, last_frame_path=f2,
                                      aspect_ratio=aspect_ratio, resolution=resolution,
                                      generate_audio=generate_audio, upsampling=upsampling):
            self.log("Bridge generation failed.")
            FileManager.cleanup_files(state["temp_files"])
            return None

        state["bridge"] = b_name
        state["temp_files"].append(b_name)
        return state

    def bridge_stitch_preview(self, state: dict) -> Optional[str]:
        preview_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "preview_stitched", ".mp4")
        files = [state["p1"], state["bridge"], state["v2_path"]]
        if MediaProcessor.concatenate_videos(files, preview_name):
            state["preview"] = preview_name
            self.log(f"Preview ready: {preview_name}")
            return preview_name
        self.log("Stitching failed.")
        return None

    def bridge_regen_v2(self, state: dict, new_v2_prompt: str) -> Optional[str]:
        """Regenerate Video 2 starting from the bridge's last frame."""
        if not new_v2_prompt:
            self.log("Please provide a prompt for the new Video 2.")
            return None

        if state.get("preview") and os.path.exists(state["preview"]):
            try:
                os.remove(state["preview"])
            except OSError as e:
                logger.warning("Could not remove old preview: %s", e)
            state["preview"] = None

        if state["v2_type"] == "generated" and state.get("v2_path") and os.path.exists(state["v2_path"]):
            try:
                os.remove(state["v2_path"])
            except OSError as e:
                logger.warning("Could not remove old V2: %s", e)

        bridge_end_frame = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "bridge_last_frame", ".png")
        state["temp_files"].append(bridge_end_frame)

        if not MediaProcessor.extract_frame(state["bridge"], bridge_end_frame, "last"):
            self.log("Bridge frame extraction failed.")
            return None

        new_v2 = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "v2_regen", ".mp4")
        v = self.ai.generate_video(new_v2_prompt, new_v2, image_path=bridge_end_frame,
                                   aspect_ratio=state.get("aspect_ratio", "16:9"),
                                   resolution=state.get("resolution", "720p"),
                                   generate_audio=state.get("generate_audio", True),
                                   upsampling=state.get("upsampling"))
        if not v:
            self.log("Regeneration failed; keeping previous Video 2.")
            return None
        state["v2_path"] = new_v2
        state["v2_type"] = "generated"
        state["temp_files"].append(new_v2)
        return new_v2

    def bridge_accept(self, state: dict, dest: Optional[str] = None) -> Optional[str]:
        """Move the accepted preview to its final destination and clean temps."""
        start = time.time()
        if not state.get("preview") or not os.path.exists(state["preview"]):
            self.log("No preview available to save.")
            return None
        if dest is None:
            dest = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "final_bridged_video", ".mp4")
        try:
            shutil.move(state["preview"], dest)
            self.log(f"Saved as {dest}")
        except OSError as e:
            self.log(f"Move failed ({e}); copying instead.")
            try:
                shutil.copy(state["preview"], dest)
            except OSError as e2:
                self.log(f"Copy failed too: {e2}")
                return None
        FileManager.cleanup_files(state.get("temp_files", []))
        if state.get("preview") and os.path.exists(state["preview"]):
            try:
                os.remove(state["preview"])
            except OSError:
                pass
        self.ai._journal.record("bridge", "ok", output=dest, duration_s=time.time() - start)
        return dest

    @staticmethod
    def open_in_player(path: str) -> None:
        """Open a media file with the OS default player (item 12)."""
        if not path or not os.path.exists(path):
            raise FileNotFoundError(path)
        opener = {"win32": "start", "darwin": "open"}.get(sys.platform, "xdg-open")
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen([opener, path])  # noqa: S603


# --- Main Application Controller (CLI) ---


class VeoGeneratorApp:
    """Interactive terminal frontend built on top of the orchestrator."""

    def __init__(self, orchestrator: Optional[GenerationOrchestrator] = None):
        Config.ensure_dirs()
        self.orch = orchestrator or GenerationOrchestrator()
        self.running = True

    @staticmethod
    def show_tasks(limit: int = 20, as_json: bool = False) -> int:
        """Print the task journal (``python veo_generator.py tasks``)."""
        Config.ensure_dirs()
        j = TaskJournal(Config.TASKS_JOURNAL_FILE)
        if as_json:
            print(json.dumps({"stats": j.stats(), "recent": j.load(limit)}, ensure_ascii=False, indent=2))
            return 0
        st = j.stats()
        print(f"Journal file: {j.path}")
        print(f"Total {st['total']} | ok {st['ok']} | failed {st['failed']} | avg {st['avg_duration_s']}s")
        for kind, counters in sorted(st["by_kind"].items()):
            print(f"  {kind:12s} ok={counters.get('ok', 0)} failed={counters.get('failed', 0)}")
        entries = j.load(limit)
        if not entries:
            print("(no tasks recorded yet)")
        else:
            print(f"\nLast {len(entries)} task(s):")
            for e in entries:
                line = f"[{e.get('ts')}] {e.get('kind'):8s} {e.get('status'):7s} {e.get('duration_s', 0)}s"
                if e.get("prompt"):
                    line += f" | {e['prompt'][:60]}"
                if e.get("output"):
                    line += f" -> {os.path.basename(e['output'])}"
                if e.get("error"):
                    line += f" | ERROR: {e['error'][:80]}"
                print(line)
        return 0

    def run(self) -> None:
        while self.running:
            try:
                self._menu_once()
            except KeyboardInterrupt:
                print("\nInterrupted by user. Returning to menu...")
            except EOFError:
                print("\nNo more input; exiting.")
                self.running = False

    def _menu_once(self) -> None:
        print("\n=== Veo3 Pro Generator ===")
        print("1. Single video (Text-to-Video)")
        print("2. Video chain (concatenated)")
        print("3. Image generation (Text-to-Image)")
        print("4. Batch mode")
        print("5. Audio generation (Lyria)")
        print("6. Bridge between videos (Interpolation with Verification)")
        print("7. Resume interrupted chain")
        print("8. Show prompt history")
        print("9. Task journal / stats")
        print("0. Exit")

        choice = input("Select mode: ")

        if choice == "0":
            self.running = False
        elif choice == "1":
            self.handle_single_video()
        elif choice == "2":
            self.handle_chain_mode()
        elif choice == "3":
            self.handle_image_gen()
        elif choice == "4":
            self.handle_batch_mode()
        elif choice == "5":
            self.handle_audio_gen()
        elif choice == "6":
            self.handle_bridge_mode()
        elif choice == "7":
            print(self.orch.resume_chain() or "Nothing to resume.")
        elif choice == "8":
            for h in FileManager.load_history(20):
                print(f"[{h['ts']}] ({h['kind']}) {h['prompt'][:80]}")
        elif choice == "9":
            self.show_task_stats()

    def show_task_stats(self) -> None:
        j = self.orch.ai._journal
        st = j.stats()
        print(f"Journal: {j.path}")
        print(f"Total {st['total']} | ok {st['ok']} | failed {st['failed']} "
              f"| avg duration {st['avg_duration_s']}s")
        for kind, counters in sorted(st["by_kind"].items()):
            print(f"  {kind:12s} ok={counters.get('ok', 0)} failed={counters.get('failed', 0)}")
        recent = j.load(10)
        if recent:
            print("Recent tasks:")
            for e in recent:
                err = f" ERROR: {e['error'][:60]}" if e.get("error") else ""
                print(f"  [{e.get('ts')}] ({e.get('kind')}) {e.get('status')} "
                      f"{e.get('duration_s', 0)}s{err}")

    def _ask_audio_upscale(self) -> tuple[bool, Optional[bool]]:
        """Ask about generated audio and native upsampling (single video flow)."""
        a = input("Generate audio in video? (Y/n): ").strip().lower()
        generate_audio = a not in ("n", "no")
        u = input("Upscale to 1080p via API (Veo 3.1)? (y/N): ").strip().lower()
        return generate_audio, (True if u in ("y", "yes") else None)

    def _ask_resolution(self) -> str:
        # Official docs: veo-3.1-generate-001 supports 720p/1080p only;
        # 4k requires the deprecated veo-3.1-generate-preview model.
        print("Resolution:")
        print("1. 720p (fast, less quota)")
        print("2. 1080p (HD)")
        if "preview" in Config.MODEL_VIDEO:
            print("3. 4k (UHD, preview model only)")
        c = input("Choice (1): ")
        if c == "2":
            return "1080p"
        if c == "3" and "preview" in Config.MODEL_VIDEO:
            return "4k"
        return "720p"

    def _configure_advanced_options(self, prompt: str) -> tuple[Optional[str], list[str], Optional[str]]:
        neg: Optional[str] = None
        if Config.GENERATE_NEGATIVE_PROMPT:
            neg = input("Negative Prompt (Enter for auto-generation): ").strip() or None
            if not neg:
                print("Generating negative prompt...")
                neg = self.orch.ai.generate_negative_prompt(prompt)
                print(f"Using negative prompt: {neg}")

        refs: list[str] = []
        c_ref = input("Add reference images? (y/n): ").lower()
        if c_ref == "y":
            print("Enter image paths (max 3); empty line to finish.")
            print("Tip: prefix a path with 'style:' or 'asset:' to force its role.")
            while len(refs) < 3:
                p_ref = input(f"Image {len(refs) + 1}: ").strip('"').strip()
                if not p_ref:
                    break
                # 'style:/path' or 'asset:C:\path' -> check the path part after the role prefix
                low = p_ref.lower()
                candidate = p_ref
                for role in ("style:", "asset:"):
                    if low.startswith(role):
                        rest = p_ref[len(role):]
                        # a Windows drive letter (e.g. 'asset:D:\\x.png') still has ':' at pos 2
                        if len(rest) > 1 and rest[1:2] == ":":
                            candidate = rest          # 'D:\x.png' - keep as-is, no strip needed
                        else:
                            candidate = rest.strip()
                        break
                if os.path.exists(candidate):
                    refs.append(p_ref)
                else:
                    print("File not found.")

        last_frame = None
        c_last = input("Set Last Frame (ending)? (1-File, 2-Generate, Enter-No): ")
        if c_last == "1":
            p_last = input("File path: ").strip('"').strip()
            if os.path.exists(p_last):
                last_frame = p_last
        elif c_last == "2":
            p_gen = input("Prompt to generate the ending frame: ")
            fname = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "gen_last_frame", ".png")
            last_frame = self.orch.ai.generate_image(p_gen, fname)

        return neg, refs, last_frame

    def handle_single_video(self) -> None:
        prompt = input("Prompt: ")
        fname = FileManager.ask_for_filename("File name (Enter for auto): ", "veo_video", Config.OUTPUT_DIRS["video"], ".mp4")
        ar = input("Format (1 - 16:9, 2 - 9:16): ")
        ratio = "9:16" if ar == "2" else "16:9"
        res = self._ask_resolution()
        dur = input("Duration seconds (Enter = default 8): ").strip()
        duration = int(dur) if dur.isdigit() else None

        neg, refs, last_frame = self._configure_advanced_options(prompt)
        generate_audio, upsampling = self._ask_audio_upscale()

        self.orch.single_video(prompt, fname, aspect_ratio=ratio, resolution=res,
                               duration_seconds=duration, negative_prompt=neg,
                               reference_image_paths=refs, last_frame_path=last_frame,
                               generate_audio=generate_audio, upsampling=upsampling)

    def handle_image_gen(self) -> None:
        p = input("Prompt: ")
        fname = FileManager.ask_for_filename("File name: ", "generated_image", Config.OUTPUT_DIRS["image"], ".png")
        self.orch.ai.generate_image(p, fname)

    def handle_audio_gen(self) -> None:
        p = input("Prompt: ")
        sec = 10
        raw = input("Seconds (10): ")
        if raw.strip():
            try:
                sec = int(raw)
            except ValueError:
                print("Invalid number; using 10.")
        fname = FileManager.ask_for_filename("File name: ", "audio", Config.OUTPUT_DIRS["audio"], ".wav")
        self.orch.ai.generate_audio(p, sec, fname)

    def handle_batch_mode(self) -> None:
        print("1. From prompt list\n2. Concatenate existing videos")
        sub = input("Choice: ")

        if sub == "1":
            raw = input("Prompts separated by ';': ")
            prompts = [x.strip() for x in raw.split(";") if x.strip()]
            res = self._ask_resolution()
            generate_audio, upsampling = self._ask_audio_upscale()
            self.orch.batch_generate(prompts, resolution=res,
                                     generate_audio=generate_audio, upsampling=upsampling)

        elif sub == "2":
            print(f"Videos must be located in: {Config.OUTPUT_DIRS['video']}")
            names = input("File names (without .mp4), comma-separated: ")
            files = []
            for n in names.split(","):
                n = n.strip()
                path = os.path.join(Config.OUTPUT_DIRS["video"], n + ".mp4")
                if os.path.exists(path):
                    files.append(path)
                else:
                    print(f"Not found: {n}")

            if len(files) > 1:
                out = FileManager.ask_for_filename("Final file name: ", "batch_concat", Config.OUTPUT_DIRS["video"], ".mp4")
                MediaProcessor.concatenate_videos(files, out)

    def handle_chain_mode(self) -> None:
        print("\n--- Chain mode ---")
        ar = input("Format for the whole chain (1 - 16:9, 2 - 9:16): ")
        ratio = "9:16" if ar == "2" else "16:9"
        res = self._ask_resolution()

        prompts: list[str] = [input("Prompt for video 1: ")]
        extra = input("How many additional links? ")
        n = int(extra) if extra.strip().isdigit() else 0
        for _ in range(n):
            prompts.append(input("Next prompt (Enter = continue action): ") or "Continue the action, high quality")

        start_image = None
        start_opt = input("Start with own image? (y/n): ").lower()
        if start_opt == "y":
            p = input("Path to image: ").strip('"')
            if os.path.exists(p):
                start_image = p

        final_name = FileManager.ask_for_filename("Final video name: ", "full_chain", Config.OUTPUT_DIRS["video"], ".mp4")
        out = self.orch.run_chain(prompts, aspect_ratio=ratio, resolution=res,
                                  start_image=start_image, final_name=final_name)
        print(f"Result: {out}" if out else "Chain interrupted (use 'Resume' from menu).")

    def handle_bridge_mode(self) -> None:
        print("\n--- Interpolation / Bridge with Verification ---")
        print(f"Video 1 name from folder {Config.OUTPUT_DIRS['video']}")
        n1 = input("Video 1: ").strip()
        p1 = os.path.join(Config.OUTPUT_DIRS["video"], n1 if n1.endswith(".mp4") else n1 + ".mp4")
        if not os.path.exists(p1):
            print("Video 1 not found.")
            return

        ar = input("Format (1 - 16:9, 2 - 9:16): ")
        ratio = "9:16" if ar == "2" else "16:9"
        res = self._ask_resolution()

        print("2. Video 2 (End):\n   1. Existing video\n   2. Generate new")
        v2_choice = input("Choice: ")
        v2_existing: Optional[str] = None
        v2_prompt: Optional[str] = None
        if v2_choice == "1":
            n2 = input("Video 2: ").strip()
            v2_existing = os.path.join(Config.OUTPUT_DIRS["video"], n2 if n2.endswith(".mp4") else n2 + ".mp4")
            if not os.path.exists(v2_existing):
                print("Video 2 not found.")
                return
        elif v2_choice == "2":
            v2_prompt = input("Prompt for Video 2: ")
        else:
            return

        prompt_bridge = input("Transition prompt (Enter for auto): ") or None
        generate_audio, upsampling = self._ask_audio_upscale()

        state = self.orch.bridge_step1_prepare(p1, v2_existing, v2_prompt, prompt_bridge,
                                               aspect_ratio=ratio, resolution=res,
                                               generate_audio=generate_audio,
                                               upsampling=upsampling)
        if not state:
            return

        while True:
            preview = self.orch.bridge_stitch_preview(state)
            if not preview:
                break
            print(f"\nDone! Check the video: {preview}")
            print("1. Yes, save")
            print("2. No, regenerate Video 2 (using bridge end frame)")
            dec = input("Choice: ")
            if dec == "1":
                final_name = FileManager.ask_for_filename("Final file name: ", "final_bridged_video",
                                                          Config.OUTPUT_DIRS["video"], ".mp4")
                self.orch.bridge_accept(state, final_name)
                break
            elif dec == "2":
                p_new = input("Prompt for new Video 2: ")
                if self.orch.bridge_regen_v2(state, p_new):
                    continue
                print("Keeping previous result; choose again.")
            else:
                print("Invalid input.")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("veo")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    try:
        os.makedirs(os.path.join(Config.BASE_DIR, "output"), exist_ok=True)
        fh = logging.FileHandler(os.path.join(Config.BASE_DIR, "output", "veo.log"), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        root.addHandler(fh)
    except OSError:
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veo-generator", description="Veo3 Pro Generator CLI")
    parser.add_argument("--interactive", "-i", action="store_true", help="force interactive menu")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command")

    vid = sub.add_parser("video", help="one-shot video generation")
    vid.add_argument("prompt")
    vid.add_argument("--output")
    vid.add_argument("--resolution", default="720p", choices=["720p", "1080p", "4k"],
                     help="4k requires VEO_MODEL_VIDEO=veo-3.1-generate-preview")
    vid.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16"])
    vid.add_argument("--duration", type=int, default=None,
                     help="clip length in whole seconds, 4-60 (Veo 3.1 docs)")
    vid.add_argument("--image", default=None, help="start-frame image path")
    vid.add_argument("--last-frame", default=None)
    vid.add_argument("--negative-prompt", default=None)
    vid.add_argument("--no-auto-negative", action="store_true", help="skip LLM negative-prompt call (saves quota)")
    vid.add_argument("--template", default=None, help="apply a named prompt template")
    vid.add_argument("--expand", action="store_true", help="expand prompt via LLM before generating")
    vid.add_argument("--no-audio", dest="audio", action="store_false", default=True,
                     help="do not generate audio track inside the video")
    vid.add_argument("--upscale", dest="upsampling", action="store_true", default=None,
                     help="ask Veo 3.1 API to upscale output to 1080p (env VEO_UPSCALE=true for default)")

    img = sub.add_parser("image", help="one-shot image generation")
    img.add_argument("prompt")
    img.add_argument("--output")

    aud = sub.add_parser("audio", help="one-shot music generation")
    aud.add_argument("prompt")
    aud.add_argument("--duration", type=int, default=10)
    aud.add_argument("--output")

    bat = sub.add_parser("batch", help="concurrent batch from a file (one prompt per line)")
    bat.add_argument("prompts_file")
    bat.add_argument("--resolution", default="720p")
    bat.add_argument("--workers", type=int, default=None)
    bat.add_argument("--no-audio", dest="audio", action="store_false", default=True)
    bat.add_argument("--upscale", dest="upsampling", action="store_true", default=None)

    tsk = sub.add_parser("tasks", help="show task journal / statistics")
    tsk.add_argument("--limit", type=int, default=20, help="how many recent tasks to print")
    tsk.add_argument("--json", action="store_true", help="machine-readable output")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    try:
        validate_api_key()
    except ApiKeyError as e:
        # 'tasks' is read-only reporting from the local journal — it must
        # work even without an API key configured.
        if args.command == "tasks":
            return VeoGeneratorApp.show_tasks(limit=args.limit, as_json=args.json)
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    if args.command == "tasks":
        return VeoGeneratorApp.show_tasks(limit=args.limit, as_json=args.json)

    Config.ensure_dirs()
    orch = GenerationOrchestrator()

    if args.command == "video":
        out = orch.single_video(
            args.prompt, args.output,
            aspect_ratio=args.aspect_ratio, resolution=args.resolution,
            duration_seconds=args.duration, image_path=args.image,
            last_frame_path=args.last_frame, negative_prompt=args.negative_prompt,
            auto_negative=not args.no_auto_negative,
            template=args.template, expand=args.expand,
            generate_audio=args.audio, upsampling=args.upsampling,
        )
        return 0 if out else 1
    if args.command == "image":
        fname = args.output or FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "image", ".png")
        return 0 if orch.ai.generate_image(args.prompt, fname) else 1
    if args.command == "audio":
        fname = args.output or FileManager.get_unique_filename(Config.OUTPUT_DIRS["audio"], "music", ".wav")
        return 0 if orch.ai.generate_audio(args.prompt, args.duration, fname) else 1
    if args.command == "batch":
        try:
            with open(args.prompts_file, "r", encoding="utf-8") as fh:
                prompts = [ln.strip() for ln in fh if ln.strip()]
        except OSError as e:
            print(f"Cannot read prompts file: {e}", file=sys.stderr)
            return 2
        results = orch.batch_generate(prompts, resolution=args.resolution, max_workers=args.workers,
                                      generate_audio=args.audio, upsampling=args.upsampling)
        return 0 if all(results) else 1

    # Default: interactive menu
    app = VeoGeneratorApp(orch)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nExiting on Ctrl+C; temporary files left in output/.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
