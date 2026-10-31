"""Flet GUI for Veo3 Pro Generator (migrated from Tkinter).

Design notes:
    * No ``sys.stdout`` monkey-patching. Worker threads push messages into a
      ``queue.Queue``; a pump thread drains it and batches UI updates —
      thread-safe with concurrent batch workers.
    * All high-level workflows are delegated to :class:`GenerationOrchestrator`
      so CLI and GUI share one implementation.
    * Shared generation options live in per-tab controls that are read at
      run time (Single / Batch / Bridge all expose the same option row).
    * Flet v1 API: ``ft.run`` + ``Page.run_thread`` for blocking work.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from typing import Callable, Optional

import flet as ft

from veo_generator import (
    ApiKeyError,
    Config,
    FileManager,
    GenerationOrchestrator,
    MediaProcessor,
    validate_api_key,
)


def _open_in_os_player(path: str) -> None:
    """Open a media file with the OS default handler."""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class OptionsRow:
    """Shared resolution/aspect/audio/upscale controls for one tab."""

    def __init__(self) -> None:
        # 4k is only offered for the preview model (official Veo 3.1 docs).
        values = ["720p", "1080p"] + (["4k"] if "preview" in Config.MODEL_VIDEO else [])
        self.res_dd = ft.Dropdown(label="Resolution", value="720p",
                                  options=[ft.dropdown.Option(v) for v in values],
                                  width=150)
        self.ar_dd = ft.Dropdown(label="Aspect Ratio", value="16:9",
                                 options=[ft.dropdown.Option("16:9"),
                                          ft.dropdown.Option("9:16")], width=150)
        self.audio_cb = ft.Checkbox(value=True, label="Audio")
        self.upscale_cb = ft.Checkbox(value=bool(Config.UPSCALE_DEFAULT),
                                      label="Upscale 1080p")
        self.row = ft.Row([self.res_dd, self.ar_dd, self.audio_cb, self.upscale_cb],
                          wrap=True, spacing=12)

    @property
    def resolution(self) -> str:
        return self.res_dd.value or "720p"

    @property
    def aspect_ratio(self) -> str:
        return self.ar_dd.value or "16:9"

    @property
    def audio(self) -> bool:
        return bool(self.audio_cb.value)

    @property
    def upscale(self) -> bool:
        return bool(self.upscale_cb.value)


class VeoGUIApp:
    def __init__(self, page: ft.Page):
        self.page = page
        page.title = "Veo3 Pro Generator UI"
        page.window.width = 980
        page.window.height = 860
        page.padding = 12

        # --- API key validation up front ---
        try:
            validate_api_key()
        except ApiKeyError as e:
            page.add(ft.Text(f"Configuration error: {e}",
                             color=ft.Colors.RED_400, size=16))
            return

        # Log-message queue drained by the pump thread (worker-safe).
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._stop_polling = threading.Event()
        self._busy = False
        self._last_video: Optional[str] = None
        self._last_image: Optional[str] = None
        self._last_audio: Optional[str] = None
        self._history_rows: list[dict] = []

        # Orchestrator routes progress lines into our queue;
        # progress_cb feeds the status-bar progress bar (percent, stage).
        self.orch = GenerationOrchestrator(log_fn=self._enqueue_log,
                                           progress_cb=self._on_progress)
        self.ai = self.orch.ai
        Config.ensure_dirs()

        # Bridge Mode State
        self.bridge_state: dict = {
            "p1": None, "bridge": None, "v2_type": None, "v2_path": None,
            "preview": None, "temp_files": [],
            "aspect_ratio": "16:9", "resolution": "720p",
        }

        self._build_ui()
        page.add(self.root_view)

        self._pump_thread = threading.Thread(target=self._pump, daemon=True)
        self._pump_thread.start()

    # ------------------------------------------------------------------ #
    # Threading plumbing
    # ------------------------------------------------------------------ #

    def _enqueue_log(self, message: str) -> None:
        """Called from worker threads; safe because it only touches a Queue."""
        self._log_queue.put(str(message))

    def _pump(self) -> None:
        """Background loop: drain log/progress state and push UI updates."""
        while not self._stop_polling.wait(0.1):
            dirty_console = False
            try:
                while True:
                    self.console.value += self._log_queue.get_nowait().rstrip("\n") + "\n"
                    dirty_console = True
                    if len(self.console.value) > 200_000:  # bound memory
                        self.console.value = self.console.value[-150_000:]
            except queue.Empty:
                pass
            if dirty_console or self._prog_dirty or self._busy_dirty:
                try:
                    if dirty_console:
                        self.console.update()
                    if self._prog_dirty:
                        self.progress_bar.update()
                        self.progress_label.update()
                        self._prog_dirty = False
                    if self._busy_dirty:
                        self.cancel_btn.update()
                        self._busy_dirty = False
                except Exception:  # noqa: BLE001 - page may be closing
                    break

    def _on_progress(self, pct: Optional[int], stage: str = "") -> None:
        """progress_cb from worker threads — mutate state; pump repaints."""
        if pct is None:  # indeterminate (waiting on API, unknown ETA)
            self.progress_bar.value = None
            self.progress_label.value = stage or "working..."
        else:
            self.progress_bar.value = max(0.0, min(1.0, pct / 100.0))
            self.progress_label.value = f"{stage} {pct}%" if stage else f"{pct}%"
        self._prog_dirty = True

    def _set_busy(self, busy: bool) -> None:
        """Toggle the Cancel button / progress state around a long task."""
        self._busy = busy
        if busy:
            self.progress_bar.value = None
            self.cancel_btn.disabled = False
        else:
            self.progress_bar.value = 0.0
            self.progress_label.value = "Idle"
            self.cancel_btn.disabled = True
        self._prog_dirty = True
        self._busy_dirty = True

    def _cancel_current(self, e=None) -> None:
        self.orch.cancel()
        self._enqueue_log("Cancellation requested — finishing current API call, then stopping...")

    def _run_in_thread(self, target: Callable[[], None]) -> None:
        def wrapper() -> None:
            self.ai.reset_cancel()
            self._set_busy(True)
            try:
                target()
            except Exception as e:  # noqa: BLE001 - never let a worker die silently
                self._enqueue_log(f"Unexpected error in worker thread: {e}")
            finally:
                self._set_busy(False)

        threading.Thread(target=wrapper, daemon=True).start()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # Status bar: progress + cancel
        self._prog_dirty = False
        self._busy_dirty = False
        self.progress_bar = ft.ProgressBar(value=0.0, width=320)
        self.progress_label = ft.Text("Idle", width=240)
        self.cancel_btn = ft.FilledTonalButton("Cancel", disabled=True,
                                               on_click=self._cancel_current)
        status_row = ft.Row(
            [self.progress_bar, self.progress_label, self.cancel_btn],
            spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # Console
        self.console = ft.TextField(
            multiline=True, min_lines=8, max_lines=12, read_only=True,
            value="", text_size=12, keyboard_type=ft.KeyboardType.NONE,
            bgcolor=ft.Colors.GREY_900, color=ft.Colors.LIGHT_GREEN_ACCENT,
        )
        console_col = ft.Column(
            [ft.Text("Log Output", weight=ft.FontWeight.BOLD), self.console],
            spacing=4,
        )

        # Tab strip: SegmentedButton selector + Stack of tab panels.
        # (ft.Tabs exists in Flet 1.x too, but the Material-3 segmented
        # control is the recommended replacement and keeps all panels
        # mounted — no state loss when switching.)
        self._tab_order = ["single", "bridge", "batch", "utils", "journal"]
        self._tab_titles = {
            "single": "Single Gen",
            "bridge": "Bridge / Interpolation",
            "batch": "Batch & Chain",
            "utils": "Image & Audio",
            "journal": "Tasks",
        }
        self._tab_panels: dict[str, ft.Control] = {
            "single": self._build_single_tab(),
            "bridge": self._build_bridge_tab(),
            "batch": self._build_batch_tab(),
            "utils": self._build_utils_tab(),
            "journal": self._build_journal_tab(),
        }
        self.tab_bar = ft.SegmentedButton(
            allow_multiple_selection=False,
            show_selected_icon=False,
            selected=["single"],
            segments=[
                ft.Segment(value=key, label=ft.Text(self._tab_titles[key]))
                for key in self._tab_order
            ],
            on_change=self._on_tab_change,
        )
        # Keep every panel mounted; toggle visibility on selection.
        self._panel_containers = {
            key: ft.Container(self._tab_panels[key],
                              visible=(key == "single"), padding=4)
            for key in self._tab_order
        }
        self.tab_stack = ft.Stack(controls=list(self._panel_containers.values()),
                                  expand=True)

        self.root_view = ft.Column(
            [self.tab_bar, self.tab_stack, console_col, ft.Divider(), status_row],
            expand=True, spacing=10,
        )

    def _switch_tab(self, key: str) -> None:
        """Show one tab panel, hide the rest (all stay mounted)."""
        for k, cont in self._panel_containers.items():
            cont.visible = (k == key)
        if key not in self.tab_bar.selected:
            self.tab_bar.selected = [key]
        try:
            self.tab_bar.update()
            self.tab_stack.update()
        except Exception:  # noqa: BLE001 - page may be closing
            pass

    def _on_tab_change(self, e: ft.ControlEvent) -> None:
        sel = e.control.selected
        if sel:
            self._switch_tab(sel[0])

    # ----- Single tab ---------------------------------------------------- #

    def _build_single_tab(self) -> ft.Container:
        self.single_prompt = ft.TextField(label="Prompt", multiline=True, min_lines=4)
        self.neg_prompt = ft.TextField(label="Negative Prompt (Auto if empty)")
        self.auto_neg_cb = ft.Checkbox(
            value=True, label="Auto-generate negative prompt (uses extra LLM quota)")
        self.chk_expand = ft.Checkbox(value=False, label="AI-expand prompt")
        self.dur_var = ft.TextField(label="Duration (s)", width=140,
                                    hint_text=f"{Config.MIN_DURATION_S}-{Config.MAX_DURATION_S}")
        self.single_opts = OptionsRow()

        self.thumb_img = ft.Image(src="", width=320, visible=False,
                                  fit=ft.BoxFit.CONTAIN)
        self.thumb_text = ft.Text("No result yet")
        preview = ft.Container(
            content=ft.Column([self.thumb_text, self.thumb_img],
                              horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            border=ft.Border(left=ft.BorderSide(1, ft.Colors.GREY_600), right=ft.BorderSide(1, ft.Colors.GREY_600), top=ft.BorderSide(1, ft.Colors.GREY_600), bottom=ft.BorderSide(1, ft.Colors.GREY_600)), border_radius=8,
            padding=10, alignment=ft.Alignment.CENTER, height=240,
        )

        btns = ft.Row([
            ft.FilledButton("Generate Video", on_click=self._run_single_gen),
            ft.OutlinedButton("Open Last Result",
                              on_click=lambda e: self._open_last(self._last_video)),
        ])

        return ft.Container(
            padding=15,
            content=ft.Column([
                self.single_prompt,
                self.single_opts.row,
                ft.Row([self.dur_var, self.chk_expand]),
                self.neg_prompt,
                self.auto_neg_cb,
                btns,
                ft.Text("Result Preview", weight=ft.FontWeight.BOLD),
                preview,
            ], spacing=10),
        )

    # ----- Batch tab ----------------------------------------------------- #

    def _build_batch_tab(self) -> ft.Container:
        self.batch_text = ft.TextField(label="Prompts (one per line)",
                                       multiline=True, min_lines=6)
        self.batch_opts = OptionsRow()
        self.workers_var = ft.TextField(label="Workers", width=110, hint_text="auto")
        self.chain_img_var = ft.TextField(label="Start image for chain (optional)",
                                          read_only=True, width=420)
        img_row = ft.Row([
            self.chain_img_var,
            ft.OutlinedButton("Browse",
                              on_click=lambda e: self._pick_image(self.chain_img_var)),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)
        btns = ft.Row([
            ft.FilledButton("Run Batch (parallel)", on_click=self._run_batch),
            ft.OutlinedButton("Run Chain (frame-linked)", on_click=self._run_chain),
            ft.OutlinedButton("Resume Chain", on_click=self._resume_chain),
        ])
        return ft.Container(
            padding=15,
            content=ft.Column([
                self.batch_text, self.batch_opts.row,
                ft.Row([self.workers_var]), img_row, btns,
            ], spacing=10),
        )

    # ----- Journal tab --------------------------------------------------- #

    def _build_journal_tab(self) -> ft.Container:
        self.journal_stats = ft.TextField(multiline=True, min_lines=3, read_only=True)
        self.journal_list = ft.ListView(height=380, spacing=2)
        btn = ft.OutlinedButton("Refresh", on_click=lambda e: self._refresh_journal())
        self._refresh_journal()
        return ft.Container(padding=15, content=ft.Column(
            [btn, self.journal_stats, self.journal_list], spacing=10))

    # ----- Utils tab ----------------------------------------------------- #

    def _build_utils_tab(self) -> ft.Container:
        self.img_prompt = ft.TextField(label="Image prompt", width=420)
        self.aud_prompt = ft.TextField(label="Audio prompt", width=420)
        self.aud_dur = ft.TextField(label="Duration (sec)", value="10", width=140)
        self.hist_list = ft.ListView(height=200, spacing=2)

        img_row = ft.Row([
            self.img_prompt,
            ft.FilledButton("Generate Image", on_click=self._run_image_gen),
            ft.OutlinedButton("Open Last",
                              on_click=lambda e: self._open_last(self._last_image)),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)
        aud_row = ft.Row([
            self.aud_prompt, self.aud_dur,
            ft.FilledButton("Generate Audio", on_click=self._run_audio_gen),
            ft.OutlinedButton("Open Last",
                              on_click=lambda e: self._open_last(self._last_audio)),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER)
        hist_btns = ft.Row([
            ft.OutlinedButton("Refresh", on_click=lambda e: self._refresh_history()),
            ft.OutlinedButton("Reuse → Single", on_click=self._reuse_history),
        ])
        self._refresh_history()
        return ft.Container(padding=15, content=ft.Column([
            ft.Text("Image Generation", weight=ft.FontWeight.BOLD), img_row,
            ft.Text("Audio Generation (Lyria)", weight=ft.FontWeight.BOLD), aud_row,
            ft.Text("Prompt History", weight=ft.FontWeight.BOLD),
            self.hist_list, hist_btns,
        ], spacing=12))

    # ----- Bridge tab ---------------------------------------------------- #

    def _build_bridge_tab(self) -> ft.Container:
        self.v1_path_var = ft.TextField(label="Video 1 (Start)", read_only=True, width=420)
        self.v2_path_var = ft.TextField(label="Video 2 (End) — optional",
                                        read_only=True, width=420)
        self.bridge_prompt = ft.TextField(label="Bridge Transition Prompt")
        self.v2_gen_prompt = ft.TextField(label="Video 2 Prompt (if generating new)")
        self.bridge_opts = OptionsRow()

        self.btn_start_bridge = ft.FilledButton("Start Bridge Process",
                                                on_click=self._start_bridge_thread)
        self.lbl_status = ft.Text("Waiting for generation...")
        self.btn_accept = ft.Button("Accept & Save", disabled=True,
                                    on_click=self._bridge_accept)
        self.btn_regen = ft.Button("Regenerate V2", disabled=True,
                                   on_click=self._bridge_regen)
        self.btn_preview = ft.Button("Play Preview", disabled=True,
                                     on_click=self._bridge_play_preview)

        def file_row(field: ft.TextField) -> ft.Row:
            return ft.Row([
                field,
                ft.OutlinedButton("Browse",
                                  on_click=lambda e, f=field: self._pick_file(f)),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER)

        return ft.Container(
            padding=15,
            content=ft.Column([
                file_row(self.v1_path_var),
                file_row(self.v2_path_var),
                ft.Text("* Leave Video 2 empty to generate it from prompt", size=11),
                self.bridge_prompt,
                self.v2_gen_prompt,
                self.bridge_opts.row,
                self.btn_start_bridge,
                ft.Container(
                    border=ft.Border(left=ft.BorderSide(1, ft.Colors.GREY_600), right=ft.BorderSide(1, ft.Colors.GREY_600), top=ft.BorderSide(1, ft.Colors.GREY_600), bottom=ft.BorderSide(1, ft.Colors.GREY_600)), border_radius=8,
                    padding=12,
                    content=ft.Column([
                        ft.Text("Verification Loop", weight=ft.FontWeight.BOLD),
                        self.lbl_status,
                        ft.Row([self.btn_accept, self.btn_regen, self.btn_preview]),
                    ], spacing=8),
                ),
            ], spacing=10),
        )

    # ------------------------------------------------------------------ #
    # File picking / previews
    # ------------------------------------------------------------------ #

    def _make_picker(self, target: ft.TextField, exts: list[str]) -> None:
        picker = ft.FilePicker()
        self.page.overlay.append(picker)
        self.page.update()

        def on_result(e: ft.FilePickerResultEvent) -> None:
            if e.files:
                target.value = e.files[0].path or e.files[0].name
                target.update()

        picker.on_result = on_result
        picker.pick_files(allow_multiple=False, dialog_title="Select file",
                          extensions=exts)

    def _pick_file(self, target: ft.TextField) -> None:
        self._make_picker(target, ["mp4", "mov", "avi", "mkv"])

    def _pick_image(self, target: ft.TextField) -> None:
        self._make_picker(target, ["png", "jpg", "jpeg", "webp"])

    def _show_thumbnail(self, video_path: str) -> None:
        """Render a first-frame preview of the result inside the GUI."""
        try:
            thumb_dir = os.path.dirname(Config.TASKS_JOURNAL_FILE)
            thumb = MediaProcessor.extract_thumbnail(
                video_path, os.path.join(thumb_dir, "gui_preview.png"))
            if not thumb:
                return
            self.thumb_img.src = thumb
            self.thumb_img.visible = True
            self.thumb_text.value = os.path.basename(video_path)
            self.thumb_img.update()
            self.thumb_text.update()
        except Exception as e:  # noqa: BLE001 - preview is strictly best-effort
            self._enqueue_log(f"Thumbnail preview failed: {e}")

    # ------------------------------------------------------------------ #
    # Journal / history
    # ------------------------------------------------------------------ #

    def _refresh_journal(self) -> None:
        """Reload the Tasks tab from the JSONL journal (instance journal path)."""
        journal = self.ai._journal
        entries = journal.load(limit=50)
        stats = journal.stats()
        controls = []
        for rec in reversed(entries):
            err = f" — {rec.get('error')}" if rec.get("error") else ""
            dur = f" {rec.get('duration_s', 0):.0f}s" if rec.get("duration_s") else ""
            line = (f"[{rec.get('ts', '')}] {str(rec.get('kind', '?')):12s} "
                    f"{str(rec.get('status', '?')):7s}{dur}{err} :: "
                    f"{str(rec.get('prompt', ''))[:50]}")
            controls.append(ft.Text(line, size=12, selectable=True))
        self.journal_list.controls = controls
        by_kind = stats.get("by_kind", {})
        lines = [f"{k}: {v.get('ok', 0)} ok / {v.get('failed', 0)} failed"
                 for k, v in sorted(by_kind.items())]
        lines.insert(0, f"total={stats.get('total', 0)}  ok={stats.get('ok', 0)}  "
                        f"failed={stats.get('failed', 0)}  "
                        f"avg={stats.get('avg_duration_s', 0.0):.1f}s")
        self.journal_stats.value = "\n".join(lines)
        try:
            self.journal_list.update()
            self.journal_stats.update()
        except Exception:  # noqa: BLE001 - during startup nothing is attached yet
            pass

    def _refresh_history(self) -> None:
        self._history_rows = FileManager.load_history(50)
        self.hist_list.controls = [
            ft.Text(f"[{h.get('ts', '')}] ({h.get('kind', '?')}) "
                    f"{str(h.get('prompt', ''))[:70]}", size=12, selectable=True)
            for h in self._history_rows
        ]

    def _reuse_history(self, e=None) -> None:
        """Copy the most recent history entry into the Single tab prompt."""
        rows = self._history_rows
        if not rows:
            self._enqueue_log("Reuse: history is empty.")
            return
        self.single_prompt.value = str(rows[-1].get("prompt", ""))
        self.single_prompt.update()
        self._switch_tab("single")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _open_last(self, path: Optional[str]) -> None:
        """Open the most recent result with the OS default player."""
        if not path or not os.path.exists(path):
            self._enqueue_log("Preview: no generated file available yet.")
            return
        try:
            _open_in_os_player(path)
        except OSError as e:
            self._enqueue_log(f"Could not open player: {e}")

    def _parse_duration(self) -> Optional[int]:
        raw = (self.dur_var.value or "").strip()
        if not raw.isdigit():
            return None
        dur = int(raw)
        lo, hi = Config.MIN_DURATION_S, Config.MAX_DURATION_S
        if not lo <= dur <= hi:
            dur = max(lo, min(dur, hi))
            self._enqueue_log(f"Duration clamped to Veo 3.1 range {lo}-{hi}s: using {dur}s.")
        return dur

    # ------------------------------------------------------------------ #
    # Actions — all flows go through the shared orchestrator
    # ------------------------------------------------------------------ #

    def _run_single_gen(self, e=None) -> None:
        prompt = (self.single_prompt.value or "").strip()
        if not prompt:
            return
        opts = self.single_opts
        neg = (self.neg_prompt.value or "").strip() or None
        auto_neg = bool(self.auto_neg_cb.value)
        expand = bool(self.chk_expand.value)
        duration = self._parse_duration()

        def task() -> None:
            self._enqueue_log("Starting Video Generation...")
            out = self.orch.single_video(
                prompt, aspect_ratio=opts.aspect_ratio, resolution=opts.resolution,
                negative_prompt=neg, auto_negative=auto_neg,
                expand=expand, duration_seconds=duration,
                generate_audio=opts.audio, upsampling=opts.upscale)
            if out:
                self._last_video = out
                self._enqueue_log(f"Done! Saved: {out}")
                self._show_thumbnail(out)
            else:
                self._enqueue_log("Generation failed or cancelled. See log above.")

        self._run_in_thread(task)

    # ---------- Batch / Chain tab ---------- #

    def _collect_batch_prompts(self) -> list[str]:
        return [ln.strip() for ln in (self.batch_text.value or "").splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    def _parse_workers(self) -> Optional[int]:
        raw = (self.workers_var.value or "").strip()
        return int(raw) if raw.isdigit() and int(raw) > 0 else None

    def _run_batch(self, e=None) -> None:
        prompts = self._collect_batch_prompts()
        if not prompts:
            self._enqueue_log("Batch: enter at least one prompt (one per line).")
            return
        opts = self.batch_opts
        workers = self._parse_workers()

        def task() -> None:
            self._enqueue_log(f"Batch: {len(prompts)} prompt(s)...")
            results = self.orch.batch_generate(
                prompts, resolution=opts.resolution, max_workers=workers,
                generate_audio=opts.audio, upsampling=opts.upscale)
            ok = sum(1 for r in results if r)
            self._enqueue_log(f"Batch finished: {ok}/{len(prompts)} succeeded.")
            if ok:
                self._last_video = next(r for r in results if r)
                self._refresh_journal()

        self._run_in_thread(task)

    def _run_chain(self, e=None) -> None:
        prompts = self._collect_batch_prompts()
        if len(prompts) < 2:
            self._enqueue_log("Chain: needs at least two prompts (one per line).")
            return
        start_img = (self.chain_img_var.value or "").strip() or None
        opts = self.batch_opts

        def task() -> None:
            self._enqueue_log(f"Chain: {len(prompts)} segment(s)...")
            out = self.orch.run_chain(
                prompts, aspect_ratio=opts.aspect_ratio,
                resolution=opts.resolution, start_image=start_img,
                generate_audio=opts.audio, upsampling=opts.upscale)
            if out:
                self._last_video = out
                self._enqueue_log(f"Chain saved: {out}")
                self._show_thumbnail(out)
            else:
                self._enqueue_log("Chain stopped/failed — use 'Resume Chain' to continue.")
            self._refresh_journal()

        self._run_in_thread(task)

    def _resume_chain(self, e=None) -> None:
        state = FileManager.load_resume_state()
        if not state:
            self._enqueue_log("Chain: no resumable chain state found.")
            return

        def task() -> None:
            out = self.orch.resume_chain()
            if out:
                self._last_video = out
                self._enqueue_log(f"Resumed chain saved: {out}")
                self._show_thumbnail(out)
            self._refresh_journal()

        self._run_in_thread(task)

    def _run_image_gen(self, e=None) -> None:
        prompt = (self.img_prompt.value or "").strip()
        if not prompt:
            return
        fname = os.path.join(Config.OUTPUT_DIRS["image"], "ui_image.png")

        def task() -> None:
            out = self.ai.generate_image(prompt, fname)
            if out:
                self._last_image = out
                self._enqueue_log(f"Image saved: {out}")

        self._run_in_thread(task)

    def _run_audio_gen(self, e=None) -> None:
        prompt = (self.aud_prompt.value or "").strip()
        if not prompt:
            return
        try:
            dur = int(self.aud_dur.value or "10")
        except ValueError:
            dur = 10
        fname = os.path.join(Config.OUTPUT_DIRS["audio"], "ui_audio.wav")

        def task() -> None:
            out = self.ai.generate_audio(prompt, dur, fname)
            if out:
                self._last_audio = out
                self._enqueue_log(f"Audio saved: {out}")

        self._run_in_thread(task)

    # ------------------------------------------------------------------ #
    # Bridge Logic (delegated to orchestrator)
    # ------------------------------------------------------------------ #

    def _start_bridge_thread(self, e=None) -> None:
        self.btn_start_bridge.disabled = True
        self.btn_accept.disabled = True
        self.btn_regen.disabled = True
        self.btn_preview.disabled = True
        self.btn_start_bridge.update()
        self._run_in_thread(self._process_bridge_initial)

    def _process_bridge_initial(self) -> None:
        opts = self.bridge_opts
        v1 = (self.v1_path_var.value or "").strip()
        v2_existing = (self.v2_path_var.value or "").strip() or None
        v2_prompt = (self.v2_gen_prompt.value or "").strip() or None
        bridge_prompt = (self.bridge_prompt.value or "").strip() or None

        state = self.orch.bridge_step1_prepare(
            v1, v2_existing, v2_prompt, bridge_prompt,
            aspect_ratio=opts.aspect_ratio, resolution=opts.resolution,
            generate_audio=opts.audio, upsampling=opts.upscale,
        )
        if not state:
            self.btn_start_bridge.disabled = False
            self.btn_start_bridge.update()
            return

        self.bridge_state = state
        preview = self.orch.bridge_stitch_preview(state)
        if preview:
            self._enable_verif_ui()
        else:
            self.btn_start_bridge.disabled = False
            self.btn_start_bridge.update()

    def _enable_verif_ui(self) -> None:
        self.lbl_status.value = (
            f"Preview created: {os.path.basename(self.bridge_state['preview'] or '')}")
        self.btn_accept.disabled = False
        self.btn_regen.disabled = False
        self.btn_preview.disabled = False
        self.btn_start_bridge.disabled = False
        try:
            self.page.update()
        except Exception:  # noqa: BLE001
            pass

    def _bridge_play_preview(self, e=None) -> None:
        self._open_last(self.bridge_state.get("preview"))

    def _bridge_accept(self, e=None) -> None:
        dest = os.path.join(Config.OUTPUT_DIRS["video"], "final_bridge.mp4")

        def task() -> None:
            out = self.orch.bridge_accept(self.bridge_state, dest)
            if out:
                self._enqueue_log(f"Saved to {out}")
                self._reset_bridge_ui()

        self._run_in_thread(task)

    def _reset_bridge_ui(self) -> None:
        self.btn_start_bridge.disabled = False
        self.btn_accept.disabled = True
        self.btn_regen.disabled = True
        self.btn_preview.disabled = True
        self.lbl_status.value = "Saved."
        try:
            self.page.update()
        except Exception:  # noqa: BLE001
            pass

    def _bridge_regen(self, e=None) -> None:
        prompt = (self.v2_gen_prompt.value or "").strip()
        if not prompt:
            self._enqueue_log("Regenerate: please enter a Video 2 prompt first.")
            return

        self.btn_accept.disabled = True
        self.btn_regen.disabled = True
        self.btn_preview.disabled = True
        self.lbl_status.value = "Regenerating V2 from bridge end..."
        self.page.update()

        self._run_in_thread(lambda: self._process_regen(prompt))

    def _process_regen(self, prompt: str) -> None:
        new_v2 = self.orch.bridge_regen_v2(self.bridge_state, prompt)
        if new_v2:
            preview = self.orch.bridge_stitch_preview(self.bridge_state)
            if preview:
                self._enable_verif_ui()
                return
        self._enqueue_log("Regeneration failed; previous state kept.")
        self._enable_verif_ui()


def main() -> None:
    # Fail fast with a readable console message instead of opening a GUI
    # window that cannot generate anything (Windows users double-click
    # run.bat, where the console stays visible).
    try:
        validate_api_key()
    except ApiKeyError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print(
            'Tip: copy .env.example to .env and set GEMINI_API_KEY '
            '(get a key at https://aistudio.google.com/apikey).',
            file=sys.stderr,
        )
        sys.exit(2)
    ft.run(VeoGUIApp)


if __name__ == "__main__":
    main()
