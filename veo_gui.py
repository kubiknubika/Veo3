"""Tkinter GUI for Veo3 Pro Generator.

Design notes:
    * No ``sys.stdout`` monkey-patching. Worker threads push messages into a
      ``queue.Queue``; the Tk main loop drains it via ``root.after`` polling —
      thread-safe UI updates (fixes interleaved writes from concurrent batch).
    * All high-level workflows are delegated to :class:`GenerationOrchestrator`
      so CLI and GUI share one implementation.
    * Shared state variables (resolution / aspect ratio) live in ``__init__``
      and are available on every tab (fixes AttributeError in Bridge/Image/Audio).
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from typing import Callable, Optional
from tkinter import ttk, filedialog, messagebox, scrolledtext

from veo_generator import (
    ApiKeyError,
    Config,
    FileManager,
    GenerationOrchestrator,
    MediaProcessor,
    TaskJournal,
    validate_api_key,
)


class VeoGUIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Veo3 Pro Generator UI")
        self.root.geometry("900x760")

        # --- API key validation up front (item 3) ---
        try:
            validate_api_key()
        except ApiKeyError as e:
            messagebox.showerror("Configuration error", str(e))
            raise SystemExit(2)

        # Log-message queue drained by the Tk main loop (item 2)
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._stop_polling = threading.Event()
        self._busy = False
        self._thumb_photo: Optional[tk.PhotoImage] = None

        # Orchestrator routes all core progress lines into our queue;
        # progress_cb feeds the status-bar progress bar (percent, stage).
        self.orch = GenerationOrchestrator(log_fn=self._enqueue_log,
                                           progress_cb=self._on_progress)
        self.ai = self.orch.ai
        Config.ensure_dirs()

        # Shared generation options — defined once, used by ALL tabs (item 1)
        self.res_var = tk.StringVar(value="720p")
        self.ar_var = tk.StringVar(value="16:9")
        self.auto_neg_var = tk.BooleanVar(value=True)
        self.upscale_var = tk.BooleanVar(value=bool(Config.UPSCALE_DEFAULT))
        self.audio_var = tk.BooleanVar(value=True)

        # Bridge Mode State
        self.bridge_state: dict = {
            "p1": None,
            "bridge": None,
            "v2_type": None,  # 'existing' or 'generated'
            "v2_path": None,
            "preview": None,
            "temp_files": [],
            "aspect_ratio": "16:9",
            "resolution": "720p",
        }

        self._init_ui()
        self._poll_log_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    # Thread-safe logging infrastructure (item 2)
    # ------------------------------------------------------------------ #

    def _enqueue_log(self, message: str) -> None:
        """Called from worker threads; safe because it only touches a Queue."""
        self._log_queue.put(str(message))

    def _poll_log_queue(self) -> None:
        """Drain the log queue from the Tk main thread."""
        if self._stop_polling.is_set():
            return
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._append_console(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _append_console(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text.rstrip("\n") + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _on_progress(self, pct: Optional[int], stage: str = "") -> None:
        """progress_cb from worker threads — marshal to the Tk main thread."""
        self._ui(lambda: self._apply_progress(pct, stage))

    def _apply_progress(self, pct: Optional[int], stage: str = "") -> None:
        if pct is None:            # indeterminate (waiting on API, unknown ETA)
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start(12)
            self.progress_label.config(text=stage or "working...")
        else:
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", value=max(0, min(100, pct)))
            self.progress_label.config(text=f"{stage} {pct}%" if stage else f"{pct}%")

    def _set_busy(self, busy: bool) -> None:
        """Toggle the Cancel button / progress animation around a long task."""
        self._busy = busy
        if busy:
            self.progress_bar.config(mode="indeterminate")
            self.progress_bar.start(12)
            self.btn_cancel.configure(state="normal")
        else:
            self.progress_bar.stop()
            self.progress_bar.config(mode="determinate", value=0)
            self.progress_label.config(text="Idle")
            self.btn_cancel.configure(state="disabled")

    def _cancel_current(self) -> None:
        self.orch.cancel()
        self._enqueue_log("Cancellation requested — finishing current API call, then stopping...")

    def _ui(self, fn: Callable[[], None]) -> None:
        """Schedule a callable on the Tk main thread."""
        self.root.after(0, fn)

    def _on_close(self) -> None:
        self._stop_polling.set()
        self.root.destroy()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Status bar (bottom): progress bar + Cancel button (items: progress/cancel)
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        self.btn_cancel = ttk.Button(status_frame, text="Cancel", command=self._cancel_current,
                                     state="disabled", width=10)
        self.btn_cancel.pack(side=tk.RIGHT, padx=5)
        self.progress_bar = ttk.Progressbar(status_frame, mode="determinate", length=320)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_label = ttk.Label(status_frame, text="Idle", width=24)
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # Output Console (above the status bar)
        console_frame = ttk.LabelFrame(main_frame, text="Log Output", height=200)
        console_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.console = scrolledtext.ScrolledText(console_frame, height=10, state="disabled", font=("Consolas", 9))
        self.console.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_single = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_single, text="Single Gen")
        self._setup_single_tab()

        self.tab_bridge = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bridge, text="Bridge / Interpolation")
        self._setup_bridge_tab()

        self.tab_batch = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_batch, text="Batch & Chain")
        self._setup_batch_tab()

        self.tab_utils = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_utils, text="Image & Audio")
        self._setup_utils_tab()

        self.tab_journal = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_journal, text="Tasks")
        self._setup_journal_tab()

    def _options_row(self, parent: ttk.Frame, row: int) -> None:
        """Shared generation options bound to app-level Vars, used on every tab."""
        # 4k is only offered for the preview model (official Veo 3.1 docs).
        res_values = ["720p", "1080p"] + (["4k"] if "preview" in Config.MODEL_VIDEO else [])
        if self.res_var.get() not in res_values:
            self.res_var.set("720p")
        ttk.Label(parent, text="Resolution:").grid(row=row, column=0, padx=5, sticky=tk.W)
        ttk.Combobox(parent, textvariable=self.res_var, values=res_values,
                     state="readonly", width=8).grid(row=row, column=1, padx=5)
        ttk.Label(parent, text="Aspect Ratio:").grid(row=row, column=2, padx=5, sticky=tk.W)
        ttk.Combobox(parent, textvariable=self.ar_var, values=["16:9", "9:16"],
                     state="readonly", width=8).grid(row=row, column=3, padx=5)
        ttk.Checkbutton(parent, text="Audio", variable=self.audio_var).grid(row=row, column=4, padx=5)
        ttk.Checkbutton(parent, text="Upscale 1080p", variable=self.upscale_var).grid(row=row, column=5, padx=5)

    def _setup_single_tab(self) -> None:
        frame = ttk.Frame(self.tab_single, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Prompt:").pack(anchor=tk.W)
        self.single_prompt = tk.Text(frame, height=5, width=60)
        self.single_prompt.pack(fill=tk.X, pady=5)

        opts = ttk.Frame(frame)
        opts.pack(fill=tk.X, pady=10)
        self._options_row(opts, 0)

        ttk.Label(opts, text="Duration (s):").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.dur_var = tk.StringVar(value="")
        ttk.Entry(opts, textvariable=self.dur_var, width=10).grid(row=1, column=1, padx=5, sticky=tk.W)

        self.chk_expand = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="AI-expand prompt", variable=self.chk_expand).grid(row=1, column=2, columnspan=2, padx=5, sticky=tk.W)

        ttk.Label(frame, text="Negative Prompt (Auto if empty):").pack(anchor=tk.W)
        self.neg_prompt = tk.Entry(frame)
        self.neg_prompt.pack(fill=tk.X, pady=5)

        self.auto_neg_chk = ttk.Checkbutton(frame, text="Auto-generate negative prompt (uses extra LLM quota)",
                                            variable=self.auto_neg_var)
        self.auto_neg_chk.pack(anchor=tk.W)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="Generate Video", command=self._run_single_gen).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Open Last Result", command=lambda: self._open_last(self._last_video)).pack(side=tk.LEFT, padx=5)

        # Result preview (thumbnail of the first frame — item: in-GUI preview)
        prev_frame = ttk.LabelFrame(frame, text="Result Preview", padding=5)
        prev_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.thumb_label = ttk.Label(prev_frame, text="No result yet", anchor=tk.CENTER)
        self.thumb_label.pack(fill=tk.BOTH, expand=True)

        self._last_video: Optional[str] = None

    def _setup_batch_tab(self) -> None:
        """Batch / Chain tab — exposes orchestrator flows that were CLI-only."""
        frame = ttk.Frame(self.tab_batch, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Prompts (one per line):").pack(anchor=tk.W)
        self.batch_text = tk.Text(frame, height=8, width=70)
        self.batch_text.pack(fill=tk.X, pady=5)

        bopts = ttk.Frame(frame)
        bopts.pack(fill=tk.X, pady=5)
        self._options_row(bopts, 0)
        ttk.Label(bopts, text="Workers:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.workers_var = tk.StringVar(value="")
        ttk.Entry(bopts, textvariable=self.workers_var, width=6).grid(row=1, column=1, padx=5, sticky=tk.W)

        ttk.Label(frame, text="Start image for chain (optional):").pack(anchor=tk.W, pady=(5, 0))
        img_frame = ttk.Frame(frame)
        img_frame.pack(fill=tk.X)
        self.chain_img_var = tk.StringVar()
        ttk.Entry(img_frame, textvariable=self.chain_img_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(img_frame, text="Browse", command=lambda: self._browse_image(self.chain_img_var)).pack(side=tk.LEFT)

        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=10)
        ttk.Button(btns, text="Run Batch (parallel)", command=self._run_batch).pack(side=tk.LEFT)
        ttk.Button(btns, text="Run Chain (frame-linked)", command=self._run_chain).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Resume Chain", command=self._resume_chain).pack(side=tk.LEFT, padx=5)

    def _setup_journal_tab(self) -> None:
        """Tasks tab — JSONL journal statistics and recent runs."""
        frame = ttk.Frame(self.tab_journal, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(frame)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Refresh", command=self._refresh_journal).pack(side=tk.LEFT)

        self.journal_stats = tk.Text(frame, height=4, state="disabled")
        self.journal_stats.pack(fill=tk.X, pady=5)

        self.journal_list = tk.Listbox(frame, height=18)
        self.journal_list.pack(fill=tk.BOTH, expand=True, pady=5)
        self._refresh_journal()

    def _setup_utils_tab(self) -> None:
        frame = ttk.Frame(self.tab_utils, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # Image Gen
        lf_img = ttk.LabelFrame(frame, text="Image Generation", padding=10)
        lf_img.pack(fill=tk.X, pady=10)

        ttk.Label(lf_img, text="Prompt:").pack(anchor=tk.W)
        self.img_prompt = tk.Entry(lf_img)
        self.img_prompt.pack(fill=tk.X, pady=5)
        img_btns = ttk.Frame(lf_img)
        img_btns.pack(fill=tk.X)
        ttk.Button(img_btns, text="Generate Image", command=self._run_image_gen).pack(side=tk.RIGHT)
        ttk.Button(img_btns, text="Open Last", command=lambda: self._open_last(self._last_image)).pack(side=tk.RIGHT, padx=5)
        self._last_image: Optional[str] = None

        # Audio Gen
        lf_aud = ttk.LabelFrame(frame, text="Audio Generation (Lyria)", padding=10)
        lf_aud.pack(fill=tk.X, pady=10)

        ttk.Label(lf_aud, text="Prompt:").pack(anchor=tk.W)
        self.aud_prompt = tk.Entry(lf_aud)
        self.aud_prompt.pack(fill=tk.X, pady=5)

        dur_frame = ttk.Frame(lf_aud)
        dur_frame.pack(fill=tk.X)
        ttk.Label(dur_frame, text="Duration (sec):").pack(side=tk.LEFT)
        self.aud_dur = tk.Entry(dur_frame, width=10)
        self.aud_dur.insert(0, "10")
        self.aud_dur.pack(side=tk.LEFT, padx=5)

        aud_btns = ttk.Frame(lf_aud)
        aud_btns.pack(fill=tk.X)
        ttk.Button(aud_btns, text="Generate Audio", command=self._run_audio_gen).pack(side=tk.RIGHT)
        ttk.Button(aud_btns, text="Open Last", command=lambda: self._open_last(self._last_audio)).pack(side=tk.RIGHT, padx=5)
        self._last_audio: Optional[str] = None

        # Prompt history (item 14)
        lf_hist = ttk.LabelFrame(frame, text="Prompt History", padding=10)
        lf_hist.pack(fill=tk.BOTH, expand=True, pady=10)
        self.hist_list = tk.Listbox(lf_hist, height=6)
        self.hist_list.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        ttk.Button(lf_hist, text="Refresh", command=self._refresh_history).pack(side=tk.RIGHT, anchor=tk.N)
        ttk.Button(lf_hist, text="Reuse → Single", command=self._reuse_history).pack(side=tk.RIGHT, anchor=tk.N, pady=5)

    def _setup_bridge_tab(self) -> None:
        frame = ttk.Frame(self.tab_bridge, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        v1_frame = ttk.Frame(frame)
        v1_frame.pack(fill=tk.X, pady=5)
        ttk.Label(v1_frame, text="Video 1 (Start):").pack(side=tk.LEFT)
        self.v1_path_var = tk.StringVar()
        ttk.Entry(v1_frame, textvariable=self.v1_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(v1_frame, text="Browse", command=lambda: self._browse_file(self.v1_path_var)).pack(side=tk.LEFT)

        v2_frame = ttk.Frame(frame)
        v2_frame.pack(fill=tk.X, pady=5)
        ttk.Label(v2_frame, text="Video 2 (End):").pack(side=tk.LEFT)
        self.v2_path_var = tk.StringVar()
        ttk.Entry(v2_frame, textvariable=self.v2_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(v2_frame, text="Browse (Optional)", command=lambda: self._browse_file(self.v2_path_var)).pack(side=tk.LEFT)
        ttk.Label(frame, text="* Leave Video 2 empty to generate it from prompt").pack(anchor=tk.W, padx=5)

        ttk.Label(frame, text="Bridge Transition Prompt:").pack(anchor=tk.W, pady=(10, 0))
        self.bridge_prompt = tk.Entry(frame)
        self.bridge_prompt.pack(fill=tk.X, pady=5)

        ttk.Label(frame, text="Video 2 Prompt (if generating new):").pack(anchor=tk.W, pady=(10, 0))
        self.v2_gen_prompt = tk.Entry(frame)
        self.v2_gen_prompt.pack(fill=tk.X, pady=5)

        bopts = ttk.Frame(frame)
        bopts.pack(fill=tk.X, pady=5)
        self._options_row(bopts, 0)

        act_frame = ttk.Frame(frame)
        act_frame.pack(fill=tk.X, pady=20)
        self.btn_start_bridge = ttk.Button(act_frame, text="Start Bridge Process", command=self._start_bridge_thread)
        self.btn_start_bridge.pack(side=tk.LEFT)

        self.verif_frame = ttk.LabelFrame(frame, text="Verification Loop", padding=10)
        self.verif_frame.pack(fill=tk.X, pady=20)

        self.lbl_status = ttk.Label(self.verif_frame, text="Waiting for generation...")
        self.lbl_status.pack(fill=tk.X)

        btn_verif = ttk.Frame(self.verif_frame)
        btn_verif.pack(fill=tk.X, pady=5)

        self.btn_accept = ttk.Button(btn_verif, text="Accept & Save", command=self._bridge_accept, state="disabled")
        self.btn_accept.pack(side=tk.LEFT, padx=5)

        self.btn_regen = ttk.Button(btn_verif, text="Regenerate V2", command=self._bridge_regen, state="disabled")
        self.btn_regen.pack(side=tk.LEFT, padx=5)

        self.btn_preview = ttk.Button(btn_verif, text="Play Preview", command=self._bridge_play_preview, state="disabled")
        self.btn_preview.pack(side=tk.LEFT, padx=5)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _browse_file(self, var: tk.StringVar) -> None:
        f = filedialog.askopenfilename(filetypes=[("Media files", "*.mp4 *.mov *.avi *.mkv")])
        if f:
            var.set(f)

    def _browse_image(self, var: tk.StringVar) -> None:
        f = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if f:
            var.set(f)

    def _show_thumbnail(self, video_path: str) -> None:
        """Render a first-frame preview of the result inside the GUI."""
        try:
            thumb_dir = os.path.dirname(Config.TASKS_JOURNAL_FILE)
            thumb = MediaProcessor.extract_thumbnail(
                video_path, os.path.join(thumb_dir, "gui_preview.png"))
            if not thumb:
                return

            def apply() -> None:
                try:
                    self._thumb_photo = tk.PhotoImage(file=thumb)
                    self.thumb_label.configure(image=self._thumb_photo, text="")
                except tk.TclError as e:  # e.g. unsupported PNG variant
                    self._enqueue_log(f"Thumbnail preview unavailable: {e}")

            self._ui(apply)
        except Exception as e:  # noqa: BLE001 - preview is strictly best-effort
            self._enqueue_log(f"Thumbnail preview failed: {e}")

    def _refresh_journal(self) -> None:
        """Reload the Tasks tab from the JSONL journal (instance journal path)."""
        journal = self.ai._journal
        entries = journal.load(limit=50)
        stats = journal.stats()
        self.journal_list.delete(0, tk.END)
        for rec in reversed(entries):
            err = f" — {rec.get('error')}" if rec.get("error") else ""
            dur = f" {rec.get('duration_s', 0):.0f}s" if rec.get("duration_s") else ""
            self.journal_list.insert(
                tk.END,
                f"[{rec.get('ts', '')}] {rec.get('kind', '?'):12s} {rec.get('status', '?'):7s}"
                f"{dur}{err} :: {str(rec.get('prompt', ''))[:50]}")
        self.journal_stats.configure(state="normal")
        self.journal_stats.delete("1.0", tk.END)
        by_kind = stats.get("by_kind", {})
        lines = [f"{k}: {v.get('ok', 0)} ok / {v.get('failed', 0)} failed"
                 for k, v in sorted(by_kind.items())]
        lines.insert(0, f"total={stats.get('total', 0)}  ok={stats.get('ok', 0)}  "
                        f"failed={stats.get('failed', 0)}  "
                        f"avg={stats.get('avg_duration_s', 0.0):.1f}s")
        self.journal_stats.insert("1.0", "\n".join(lines))
        self.journal_stats.configure(state="disabled")

    def _run_in_thread(self, target: Callable[[], None]) -> None:
        def wrapper() -> None:
            self.ai.reset_cancel()
            self._ui(lambda: self._set_busy(True))
            try:
                target()
            except Exception as e:  # noqa: BLE001 - never let a worker die silently
                self._enqueue_log(f"Unexpected error in worker thread: {e}")
            finally:
                self._ui(lambda: self._set_busy(False))

        threading.Thread(target=wrapper, daemon=True).start()

    def _open_last(self, path: Optional[str]) -> None:
        """Open the most recent result with the OS default player (item 12)."""
        if not path or not os.path.exists(path):
            messagebox.showinfo("Preview", "No generated file available yet.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except OSError as e:
            messagebox.showerror("Preview", f"Could not open player: {e}")

    def _refresh_history(self) -> None:
        self.hist_list.delete(0, tk.END)
        for h in FileManager.load_history(50):
            self.hist_list.insert(tk.END, f"[{h.get('ts', '')}] ({h.get('kind', '?')}) {str(h.get('prompt', ''))[:70]}")

    def _reuse_history(self) -> None:
        sel = self.hist_list.curselection()
        if not sel:
            return
        line = self.hist_list.get(sel[0])
        prompt = line.split(") ", 1)[-1]
        self.single_prompt.delete("1.0", tk.END)
        self.single_prompt.insert("1.0", prompt)
        self.notebook.select(self.tab_single)

    def _parse_duration(self) -> Optional[int]:
        raw = self.dur_var.get().strip()
        if not raw.isdigit():
            return None
        dur = int(raw)
        lo, hi = Config.MIN_DURATION_S, Config.MAX_DURATION_S
        if not lo <= dur <= hi:
            dur = max(lo, min(dur, hi))
            self._enqueue_log(f"Duration clamped to Veo 3.1 range {lo}-{hi}s: using {dur}s.")
        return dur

    # ------------------------------------------------------------------ #
    # Actions — all flows go through the shared orchestrator (item 8)
    # ------------------------------------------------------------------ #

    def _run_single_gen(self) -> None:
        prompt = self.single_prompt.get("1.0", tk.END).strip()
        if not prompt:
            return

        neg = self.neg_prompt.get().strip() or None
        ar = self.ar_var.get()
        res = self.res_var.get()
        auto_neg = self.auto_neg_var.get()
        expand = self.chk_expand.get()
        duration = self._parse_duration()

        def task() -> None:
            self._enqueue_log("Starting Video Generation...")
            out = self.orch.single_video(prompt, aspect_ratio=ar, resolution=res,
                                         negative_prompt=neg, auto_negative=auto_neg,
                                         expand=expand, duration_seconds=duration,
                                         generate_audio=self.audio_var.get(),
                                         upsampling=self.upscale_var.get())
            if out:
                self._last_video = out
                self._enqueue_log(f"Done! Saved: {out}")
                self._show_thumbnail(out)
            else:
                self._enqueue_log("Generation failed or cancelled. See log above.")

        self._run_in_thread(task)

    # ---------- Batch / Chain tab ----------

    def _collect_batch_prompts(self) -> list[str]:
        return [ln.strip() for ln in self.batch_text.get("1.0", tk.END).splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    def _parse_workers(self) -> Optional[int]:
        raw = self.workers_var.get().strip()
        return int(raw) if raw.isdigit() and int(raw) > 0 else None

    def _run_batch(self) -> None:
        prompts = self._collect_batch_prompts()
        if not prompts:
            messagebox.showinfo("Batch", "Enter at least one prompt (one per line).")
            return

        def task() -> None:
            self._enqueue_log(f"Batch: {len(prompts)} prompt(s)...")
            results = self.orch.batch_generate(
                prompts, resolution=self.res_var.get(), max_workers=self._parse_workers(),
                generate_audio=self.audio_var.get(), upsampling=self.upscale_var.get())
            ok = sum(1 for r in results if r)
            self._enqueue_log(f"Batch finished: {ok}/{len(prompts)} succeeded.")
            if ok:
                self._last_video = next(r for r in results if r)
                self._ui(lambda: self._refresh_journal())

        self._run_in_thread(task)

    def _run_chain(self) -> None:
        prompts = self._collect_batch_prompts()
        if len(prompts) < 2:
            messagebox.showinfo("Chain", "A chain needs at least two prompts (one per line).")
            return
        start_img = self.chain_img_var.get().strip() or None

        def task() -> None:
            self._enqueue_log(f"Chain: {len(prompts)} segment(s)...")
            out = self.orch.run_chain(
                prompts, aspect_ratio=self.ar_var.get(), resolution=self.res_var.get(),
                start_image=start_img,
                generate_audio=self.audio_var.get(), upsampling=self.upscale_var.get())
            if out:
                self._last_video = out
                self._enqueue_log(f"Chain saved: {out}")
                self._show_thumbnail(out)
            else:
                self._enqueue_log("Chain stopped/failed — use 'Resume Chain' to continue.")
            self._ui(lambda: self._refresh_journal())

        self._run_in_thread(task)

    def _resume_chain(self) -> None:
        state = FileManager.load_resume_state()
        if not state:
            messagebox.showinfo("Chain", "No resumable chain state found.")
            return

        def task() -> None:
            out = self.orch.resume_chain()
            if out:
                self._last_video = out
                self._enqueue_log(f"Resumed chain saved: {out}")
                self._show_thumbnail(out)
            self._ui(lambda: self._refresh_journal())

        self._run_in_thread(task)

    def _run_image_gen(self) -> None:
        prompt = self.img_prompt.get().strip()
        if not prompt:
            return

        fname = filedialog.asksaveasfilename(
            initialdir=Config.OUTPUT_DIRS["image"],
            initialfile="ui_image.png",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
        )
        if not fname:
            return

        def task() -> None:
            out = self.ai.generate_image(prompt, fname)
            if out:
                self._last_image = out
                self._enqueue_log(f"Image saved: {out}")

        self._run_in_thread(task)

    def _run_audio_gen(self) -> None:
        prompt = self.aud_prompt.get().strip()
        if not prompt:
            return
        try:
            dur = int(self.aud_dur.get())
        except ValueError:
            dur = 10

        fname = filedialog.asksaveasfilename(
            initialdir=Config.OUTPUT_DIRS["audio"],
            initialfile="ui_audio.wav",
            defaultextension=".wav",
            filetypes=[("WAV Audio", "*.wav")],
        )
        if not fname:
            return

        def task() -> None:
            out = self.ai.generate_audio(prompt, dur, fname)
            if out:
                self._last_audio = out
                self._enqueue_log(f"Audio saved: {out}")

        self._run_in_thread(task)

    # ------------------------------------------------------------------ #
    # Bridge Logic (delegated to orchestrator)
    # ------------------------------------------------------------------ #

    def _start_bridge_thread(self) -> None:
        self.btn_start_bridge.configure(state="disabled")
        self.btn_accept.configure(state="disabled")
        self.btn_regen.configure(state="disabled")
        self.btn_preview.configure(state="disabled")
        self._run_in_thread(self._process_bridge_initial)

    def _process_bridge_initial(self) -> None:
        v1 = self.v1_path_var.get().strip()
        v2_existing = self.v2_path_var.get().strip() or None
        v2_prompt = self.v2_gen_prompt.get().strip() or None
        bridge_prompt = self.bridge_prompt.get().strip() or None

        state = self.orch.bridge_step1_prepare(
            v1, v2_existing, v2_prompt, bridge_prompt,
            aspect_ratio=self.ar_var.get(), resolution=self.res_var.get(),
            generate_audio=self.audio_var.get(), upsampling=self.upscale_var.get(),
        )
        if not state:
            self._ui(lambda: self.btn_start_bridge.configure(state="normal"))
            return

        self.bridge_state = state
        preview = self.orch.bridge_stitch_preview(state)
        if preview:
            self._ui(self._enable_verif_ui)
        else:
            self._ui(lambda: self.btn_start_bridge.configure(state="normal"))

    def _enable_verif_ui(self) -> None:
        self.lbl_status.configure(text=f"Preview created: {os.path.basename(self.bridge_state['preview'] or '')}")
        self.btn_accept.configure(state="normal")
        self.btn_regen.configure(state="normal")
        self.btn_preview.configure(state="normal")

    def _bridge_play_preview(self) -> None:
        self._open_last(self.bridge_state.get("preview"))

    def _bridge_accept(self) -> None:
        dest = filedialog.asksaveasfilename(defaultextension=".mp4",
                                            initialdir=Config.OUTPUT_DIRS["video"],
                                            initialfile="final_bridge.mp4")
        if not dest:
            return

        def task() -> None:
            out = self.orch.bridge_accept(self.bridge_state, dest)
            if out:
                self._enqueue_log(f"Saved to {out}")
                self._ui(self._reset_bridge_ui)

        self._run_in_thread(task)

    def _reset_bridge_ui(self) -> None:
        self.btn_start_bridge.configure(state="normal")
        self.btn_accept.configure(state="disabled")
        self.btn_regen.configure(state="disabled")
        self.btn_preview.configure(state="disabled")
        self.lbl_status.configure(text="Saved.")

    def _bridge_regen(self) -> None:
        prompt = self.v2_gen_prompt.get().strip()
        if not prompt:
            messagebox.showinfo("Regenerate", "Please enter a Video 2 prompt for regeneration.")
            return

        self.btn_accept.configure(state="disabled")
        self.btn_regen.configure(state="disabled")
        self.btn_preview.configure(state="disabled")
        self.lbl_status.configure(text="Regenerating V2 from bridge end...")

        self._run_in_thread(lambda: self._process_regen(prompt))

    def _process_regen(self, prompt: str) -> None:
        new_v2 = self.orch.bridge_regen_v2(self.bridge_state, prompt)
        if new_v2:
            preview = self.orch.bridge_stitch_preview(self.bridge_state)
            if preview:
                self._ui(self._enable_verif_ui)
                return
        self._enqueue_log("Regeneration failed; previous state kept.")
        self._ui(self._enable_verif_ui)


def main() -> None:
    root = tk.Tk()
    try:
        VeoGUIApp(root)
    except SystemExit:
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
