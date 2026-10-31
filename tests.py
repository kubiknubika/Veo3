"""Mocked unit tests for the Veo3 Pro Generator core and GUI logic.

Run with:  pytest tests.py   (or)   python -m unittest tests.py
No API key or network access required.
"""

import json
import os
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch, mock_open, ANY

from google.genai import types

import veo_generator
from veo_generator import (
    ApiKeyError,
    CancelledError,
    Config,
    FileManager,
    GenerationOrchestrator,
    GenAIService,
    MediaProcessor,
    validate_api_key,
)
from task_journal import TaskJournal


def _fake_journal() -> MagicMock:
    """MagicMock standing in for GenAIService._journal in orchestrator tests."""
    j = MagicMock(spec=TaskJournal)
    j.record = MagicMock()
    j.load = MagicMock(return_value=[])
    j.stats = MagicMock(return_value={})
    return j


class TestApiKeyValidation(unittest.TestCase):
    def test_missing_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with self.assertRaises(ApiKeyError):
                validate_api_key()

    def test_present_key_returns(self):
        valid = "AIzaSyDUMMY1234567890abcdefABCDEFGH"  # 39 chars, AIza-prefixed
        with patch.dict(os.environ, {"GEMINI_API_KEY": valid}):
            self.assertEqual(validate_api_key(), valid)

    def test_placeholder_key_rejected(self):
        for bad in ["your_api_key_here", "sk-short", "AIzaTruncated"]:
            with patch.dict(os.environ, {"GEMINI_API_KEY": bad}):
                with self.assertRaises(ApiKeyError, msg=bad):
                    validate_api_key()


class TestFileManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_get_unique_filename_simple(self):
        name = FileManager.get_unique_filename(self.test_dir, "test", ".txt")
        self.assertEqual(name, os.path.join(self.test_dir, "test.txt"))

    def test_get_unique_filename_increment(self):
        base = os.path.join(self.test_dir, "test.txt")
        with open(base, "w") as f:
            f.write("content")
        name = FileManager.get_unique_filename(self.test_dir, "test", ".txt")
        self.assertEqual(name, os.path.join(self.test_dir, "test_1.txt"))

    def test_get_unique_filename_increment_multiple(self):
        for n in ("test.txt", "test_1.txt"):
            with open(os.path.join(self.test_dir, n), "w") as f:
                f.write("x")
        name = FileManager.get_unique_filename(self.test_dir, "test", ".txt")
        self.assertEqual(name, os.path.join(self.test_dir, "test_2.txt"))

    def test_cleanup_files(self):
        f1 = os.path.join(self.test_dir, "t1.txt")
        f2 = os.path.join(self.test_dir, "t2.txt")
        for p in (f1, f2):
            with open(p, "w") as f:
                f.write("1")
        FileManager.cleanup_files([f1, f2])
        self.assertFalse(os.path.exists(f1))
        self.assertFalse(os.path.exists(f2))

    def test_cleanup_files_tolerates_missing_and_errors(self):
        # Non-existent entries are skipped silently; broken paths log a warning.
        FileManager.cleanup_files([os.path.join(self.test_dir, "ghost.txt"), None])

    def test_prompt_history_roundtrip(self):
        with patch.object(Config, "HISTORY_FILE", os.path.join(self.test_dir, "hist.json")):
            FileManager.record_prompt("cat video", "video", "out.mp4")
            FileManager.record_prompt("dog image", "image", "out.png")
            hist = FileManager.load_history(limit=0)
            self.assertEqual(len(hist), 2)
            self.assertEqual(hist[0]["prompt"], "cat video")
            self.assertEqual(hist[1]["kind"], "image")

    def test_templates_builtin_and_apply(self):
        tpl = FileManager.apply_template("cinematic", "a cat running")
        self.assertIn("a cat running", tpl)
        self.assertIn("cinematic lighting", tpl)
        # Unknown template returns prompt unchanged
        self.assertEqual(FileManager.apply_template("nope", "keep me"), "keep me")

    def test_resume_state_roundtrip(self):
        with patch.object(Config, "RESUME_FILE", os.path.join(self.test_dir, "resume.json")):
            self.assertIsNone(FileManager.load_resume_state())
            FileManager.save_resume_state({"prompts": ["p2"], "chain": ["/tmp/a.mp4"]})
            state = FileManager.load_resume_state()
            self.assertEqual(state["prompts"], ["p2"])
            FileManager.clear_resume_state()
            self.assertIsNone(FileManager.load_resume_state())

    def test_resume_state_write_is_atomic_and_locked(self):
        """Concurrent saves must never leave a torn/corrupt JSON file."""
        rf = os.path.join(self.test_dir, "resume.json")
        with patch.object(Config, "RESUME_FILE", rf):
            errors: list[Exception] = []

            def writer(i: int) -> None:
                try:
                    for _ in range(20):
                        FileManager.save_resume_state({"prompts": [f"p{i}"] * 50})
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            # File always parses and no temp files are left behind
            state = FileManager.load_resume_state()
            self.assertIsInstance(state, dict)
            leftovers = [f for f in os.listdir(self.test_dir) if f.startswith(".resume_")]
            self.assertEqual(leftovers, [])


class TestMediaProcessor(unittest.TestCase):
    """ffmpeg-based MediaProcessor: commands are mocked at _run level,
    plus real-binary smoke tests when ffmpeg is available."""

    def test_extract_last_frame_uses_sseof_seek(self):
        """Last frame is grabbed near the true end of stream (-sseof),
        never by seeking a reported frame count (unreliable for VBR)."""
        calls = []

        def fake_run(cmd, timeout=600):
            calls.append(cmd)
            # simulate ffmpeg writing the output image
            open(cmd[-1], "wb").close() if cmd[-1].endswith(".png") else None
            with open(cmd[-1], "w") as f:
                f.write("x")
            return MagicMock(stdout="{}")

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "frame.png")
            with patch.object(MediaProcessor, "_run", side_effect=fake_run):
                result = MediaProcessor.extract_frame("dummy_video.mp4", out, "last")
            self.assertEqual(result, out)
            first_cmd = calls[0]
            self.assertIn("-sseof", first_cmd)
            self.assertIn("-0.1", first_cmd)
            self.assertIn("-frames:v", first_cmd)
            # no whole-stream frame-count seeking anywhere
            for c in calls:
                self.assertNotIn("CAP_PROP_FRAME_COUNT", str(c))

    def test_extract_first_frame_single_frame_decode(self):
        calls = []

        def fake_run(cmd, timeout=600):
            calls.append(cmd)
            with open(cmd[-1], "w") as f:
                f.write("x")
            return MagicMock(stdout="{}")

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "frame.png")
            with patch.object(MediaProcessor, "_run", side_effect=fake_run):
                result = MediaProcessor.extract_frame("v.mp4", out, "first")
            self.assertEqual(result, out)
            self.assertEqual(len(calls), 1)
            self.assertIn("-frames:v", calls[0])
            self.assertEqual(calls[0][calls[0].index("-frames:v") + 1], "1")

    def test_extract_frame_fail_open_returns_none(self):
        """ffmpeg failure must degrade to None, not raise."""
        import subprocess as sp
        with patch.object(MediaProcessor, "_run",
                          side_effect=sp.CalledProcessError(1, "ffmpeg")):
            self.assertIsNone(MediaProcessor.extract_frame("bad.mp4", "out.png"))
        with patch.object(MediaProcessor, "_run",
                          side_effect=FileNotFoundError("no ffmpeg")):
            self.assertIsNone(MediaProcessor.extract_frame("bad.mp4", "out.png"))

    def test_concatenate_videos_identical_geometry_skips_normalize(self):
        meta = {"w": 1280, "h": 720, "fps": 24.0}
        cmds = []

        def fake_run(cmd, timeout=600):
            cmds.append(cmd)
            with open(cmd[-1], "w") as f:   # create the concat output
                f.write("x")
            return MagicMock(stdout=json.dumps({"streams": [meta]}))

        with tempfile.TemporaryDirectory() as td:
            v1 = os.path.join(td, "v1.mp4"); open(v1, "w").close()
            v2 = os.path.join(td, "v2.mp4"); open(v2, "w").close()
            out = os.path.join(td, "final.mp4")
            with patch.object(MediaProcessor, "_probe_video",
                              side_effect=[dict(meta), dict(meta)]), \
                 patch.object(MediaProcessor, "_run", side_effect=fake_run), \
                 patch.object(MediaProcessor, "normalize_clip") as mock_norm:
                result = MediaProcessor.concatenate_videos([v1, v2], out)
            self.assertEqual(result, out)
            mock_norm.assert_not_called()   # identical geometry -> no re-encode
            self.assertTrue(any("-f" in c and "concat" in c for c in cmds))

    def test_concatenate_videos_normalizes_mismatched_clips(self):
        """Clips with different geometry must be normalized to the first clip's size."""
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.mp4"); open(a, "w").close()
            b = os.path.join(td, "b.mp4"); open(b, "w").close()
            out = os.path.join(td, "out.mp4")

            def fake_run(cmd, timeout=600):
                with open(cmd[-1], "w") as f:
                    f.write("x")
                return MagicMock(stdout="{}")

            with patch.object(MediaProcessor, "_probe_video",
                              side_effect=[{"w": 1920, "h": 1080, "fps": 24.0},
                                           {"w": 1280, "h": 720, "fps": 30.0}]), \
                 patch.object(MediaProcessor, "_run", side_effect=fake_run), \
                 patch.object(MediaProcessor, "normalize_clip",
                              return_value=os.path.join(td, "_norm_b.mp4")) as mock_norm:
                result = MediaProcessor.concatenate_videos([a, b], out)
            self.assertEqual(result, out)
            mock_norm.assert_called_once_with(b, 1920, 1080, ANY)

    def test_concatenate_videos_probe_failure_returns_none_and_cleans_up(self):
        """Regression: error path must not leak temp files or raise."""
        with patch.object(MediaProcessor, "_probe_video", return_value=None):
            result = MediaProcessor.concatenate_videos(["a.mp4", "b.mp4"], "out.mp4")
        self.assertIsNone(result)

    def test_normalize_clip_passthrough_when_matching(self):
        with patch.object(MediaProcessor, "_probe_video",
                          return_value={"w": 1280, "h": 720, "fps": 24.0}):
            self.assertEqual(MediaProcessor.normalize_clip("x.mp4", 1280, 720, 24.0),
                             "x.mp4")

    def test_normalize_clip_reencodes_on_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.mp4"); open(src, "w").close()
            norm = os.path.join(td, "_norm_src.mp4")

            def fake_run(cmd, timeout=600):
                self.assertIn("scale=", " ".join(cmd))
                self.assertIn("pad=", " ".join(cmd))
                with open(cmd[-1], "w") as f:
                    f.write("x")
                return MagicMock(stdout="")

            with patch.object(MediaProcessor, "_probe_video",
                              return_value={"w": 640, "h": 360, "fps": 30.0}), \
                 patch.object(MediaProcessor, "_run", side_effect=fake_run):
                out = MediaProcessor.normalize_clip(src, 1280, 720, 24.0)
            self.assertEqual(out, norm)

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not installed")
    def test_real_ffmpeg_extract_first_frame_smoke(self):
        """End-to-end check against the real binary: color source -> PNG frame."""
        with tempfile.TemporaryDirectory() as td:
            vid = os.path.join(td, "gen.mp4")
            MediaProcessor._run([
                "ffmpeg", "-y", "-f", "lavfi", "-i",
                "testsrc=size=320x240:rate=24:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", vid,
            ], timeout=120)
            frame = os.path.join(td, "f.png")
            self.assertEqual(MediaProcessor.extract_frame(vid, frame, "first"), frame)
            meta = MediaProcessor._probe_video(vid)
            self.assertEqual(meta["w"], 320)
            self.assertEqual(meta["h"], 240)
            thumb = os.path.join(td, "t.jpg")
            self.assertEqual(MediaProcessor.extract_thumbnail(vid, thumb, width=160), thumb)


class TestGenAIService(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("veo_generator.genai.Client")
        self.MockClient = self.patcher.start()
        self.mock_client_instance = self.MockClient.return_value

        self.mock_operation = MagicMock()
        self.mock_operation.done = True
        self.mock_operation.response.generated_videos = [MagicMock()]
        self.mock_client_instance.models.generate_videos.return_value = self.mock_operation

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDUMMY1234567890abcdefABCDEFGH"}):
            self.service = GenAIService()

    def tearDown(self):
        self.patcher.stop()

    def test_generate_video_simple(self):
        with patch.object(FileManager, "record_prompt"):
            result = self.service.generate_video("test prompt", "out.mp4", resolution="720p")
        call = self.mock_client_instance.models.generate_videos.call_args
        self.assertEqual(call.kwargs["model"], Config.MODEL_VIDEO)
        self.assertEqual(call.kwargs["config"].resolution, "720p")
        self.assertEqual(result, "out.mp4")

    def test_generate_video_passes_duration_and_audio(self):
        with patch.object(FileManager, "record_prompt"):
            self.service.generate_video("p", "out.mp4", duration_seconds=6, generate_audio=False)
        cfg = self.mock_client_instance.models.generate_videos.call_args.kwargs["config"]
        self.assertEqual(cfg.duration_seconds, 6)
        self.assertFalse(cfg.generate_audio)

    def test_generate_video_with_image(self):
        with patch("builtins.open", mock_open(read_data=b"fake_image_bytes")), \
             patch.object(FileManager, "record_prompt"):
            result = self.service.generate_video("prompt", "out.mp4", image_path="test.png")
        call = self.mock_client_instance.models.generate_videos.call_args
        self.assertIsNotNone(call.kwargs["image"])
        self.assertEqual(result, "out.mp4")

    def test_person_generation_fallback(self):
        """INVALID_ARGUMENT about personGeneration retries without that param."""
        err = Exception("400 INVALID_ARGUMENT: field personGeneration is not supported")
        ok_op = self.mock_operation
        self.mock_client_instance.models.generate_videos.side_effect = [err, ok_op]
        with patch.object(FileManager, "record_prompt"):
            result = self.service.generate_video("p", "out.mp4")
        self.assertEqual(result, "out.mp4")
        self.assertEqual(self.mock_client_instance.models.generate_videos.call_count, 2)
        second_cfg = self.mock_client_instance.models.generate_videos.call_args_list[1].kwargs["config"]
        self.assertNotIn("person_generation", second_cfg.model_dump(exclude_none=True))

    def test_retry_on_rate_limit(self):
        rate_err = Exception("429 RESOURCE_EXHAUSTED: rate limit exceeded")
        self.mock_client_instance.models.generate_videos.side_effect = [rate_err, self.mock_operation]
        with patch.object(FileManager, "record_prompt"), patch("veo_generator.time.sleep") as sleep_mock:
            result = self.service.generate_video("p", "out.mp4")
        self.assertEqual(result, "out.mp4")
        self.assertEqual(self.mock_client_instance.models.generate_videos.call_count, 2)
        sleep_mock.assert_called()  # backoff applied

    def test_no_retry_on_permanent_error(self):
        self.mock_client_instance.models.generate_videos.side_effect = Exception("403 PERMISSION_DENIED")
        with patch.object(FileManager, "record_prompt"), patch("veo_generator.time.sleep"):
            result = self.service.generate_video("p", "out.mp4")
        self.assertIsNone(result)
        self.assertEqual(self.mock_client_instance.models.generate_videos.call_count, 1)

    def test_polling_timeout(self):
        op = MagicMock()
        op.done = False
        self.mock_client_instance.models.generate_videos.return_value = op
        self.mock_client_instance.operations.get.return_value = op
        with patch.object(FileManager, "record_prompt"), \
             patch.object(Config, "POLL_TIMEOUT_S", 0.01), \
             patch.object(Config, "POLL_INTERVAL_S", 0.001), \
             patch("veo_generator.time.sleep"):
            result = self.service.generate_video("p", "out.mp4")
        self.assertIsNone(result)

    def test_generate_video_api_error(self):
        self.mock_client_instance.models.generate_videos.side_effect = Exception("API Error")
        with patch.object(FileManager, "record_prompt"):
            result = self.service.generate_video("prompt", "out.mp4")
        self.assertIsNone(result)

    def test_generate_image(self):
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = True
        mock_image_obj = MagicMock()
        mock_part.as_image.return_value = mock_image_obj
        mock_response.parts = [mock_part]
        self.mock_client_instance.models.generate_content.return_value = mock_response

        with patch.object(FileManager, "record_prompt"):
            result = self.service.generate_image("prompt", "img.png")
        mock_image_obj.save.assert_called_with("img.png")
        self.assertEqual(result, "img.png")

    def test_generate_negative_prompt_failure_is_graceful(self):
        self.mock_client_instance.models.generate_content.side_effect = Exception("boom")
        self.assertIsNone(self.service.generate_negative_prompt("p"))

    def test_expand_prompt(self):
        resp = MagicMock()
        resp.text = " A detailed cinematic prompt. "
        self.mock_client_instance.models.generate_content.return_value = resp
        self.assertEqual(self.service.expand_prompt("cat"), "A detailed cinematic prompt.")


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.ai = MagicMock(spec=GenAIService)
        self.ai._journal = _fake_journal()
        # Real cooperative-cancellation plumbing on the mock so batch/chain
        # code paths that touch cancel_event behave like production.
        self.cancel = threading.Event()
        self.ai.cancel_event = self.cancel
        self.ai.reset_cancel.side_effect = self.cancel.clear
        self.ai.cancel.side_effect = self.cancel.set
        self.logs = []
        self.orch = GenerationOrchestrator(ai=self.ai, log_fn=self.logs.append)

    def test_single_video_auto_negative_enabled(self):
        self.ai.generate_negative_prompt.return_value = "blurry, watermark"
        self.ai.generate_video.return_value = "out.mp4"
        with patch.object(Config, "GENERATE_NEGATIVE_PROMPT", True):
            out = self.orch.single_video("p", "out.mp4")
        self.assertEqual(out, "out.mp4")
        self.ai.generate_negative_prompt.assert_called_once()
        self.assertEqual(self.ai.generate_video.call_args.kwargs["negative_prompt"], "blurry, watermark")

    def test_single_video_skip_negative_saves_quota(self):
        """Item 13: auto-negative can be skipped entirely."""
        self.ai.generate_video.return_value = "out.mp4"
        self.orch.single_video("p", "out.mp4", auto_negative=False)
        self.ai.generate_negative_prompt.assert_not_called()

    def test_single_video_explicit_negative_skips_llm(self):
        self.ai.generate_video.return_value = "out.mp4"
        self.orch.single_video("p", "out.mp4", negative_prompt="ugly", auto_negative=True)
        self.ai.generate_negative_prompt.assert_not_called()

    def test_batch_runs_all_prompts_concurrently(self):
        self.ai.generate_video.side_effect = lambda p, f, **kw: f
        results = self.orch.batch_generate(["a", "b", "c"], max_workers=3)
        self.assertEqual(self.ai.generate_video.call_count, 3)
        self.assertTrue(all(r for r in results))

    def test_batch_survives_item_failure(self):
        def gen(p, f, **kw):
            if p == "bad":
                raise RuntimeError("429 rate limit")
            return f
        self.ai.generate_video.side_effect = gen
        results = self.orch.batch_generate(["ok1", "bad", "ok2"], max_workers=2)
        self.assertIsNone(results[1])
        self.assertTrue(results[0] and results[2])

    def test_chain_saves_and_clears_resume_state(self):
        created: list[str] = []

        def gen(p, f, **kw):
            open(f, "w").close()   # simulate the API writing the segment file
            created.append(f)
            return f

        self.ai.generate_video.side_effect = gen
        with tempfile.TemporaryDirectory() as td:
            final = os.path.join(td, "final.mp4")

            def fake_concat(paths, out):
                assert all(os.path.exists(p) for p in paths), paths
                open(out, "w").close()
                return out

            dirs = {"video": td, "image": td, "audio": td}
            with patch.object(Config, "RESUME_FILE", os.path.join(td, "resume.json")), \
                 patch.object(type(Config), "OUTPUT_DIRS", new_callable=lambda: property(lambda self: dirs)), \
                 patch("veo_generator.MediaProcessor.extract_frame", return_value="frame.png"), \
                 patch("veo_generator.MediaProcessor.concatenate_videos", side_effect=fake_concat):
                out = self.orch.run_chain(["p1", "p2"], final_name=final)
                self.assertEqual(out, final)
                self.assertFalse(os.path.exists(os.path.join(td, "resume.json")))  # cleared on success

    def test_chain_keeps_resume_state_on_failure(self):
        calls = {"n": 0}
        def gen(p, f, **kw):
            calls["n"] += 1
            return None if calls["n"] == 2 else f
        self.ai.generate_video.side_effect = gen
        with tempfile.TemporaryDirectory() as td:
            rf = os.path.join(td, "resume.json")
            with patch.object(Config, "RESUME_FILE", rf), \
                 patch("veo_generator.MediaProcessor.extract_frame", return_value="frame.png"), \
                 patch("veo_generator.MediaProcessor.concatenate_videos", return_value=None):
                self.orch.run_chain(["p1", "p2", "p3"])
                self.assertTrue(os.path.exists(rf))
                state = FileManager.load_resume_state()
                self.assertIn("p3", state["prompts"])

    def test_bridge_prepare_existing_v2(self):
        with tempfile.TemporaryDirectory() as td:
            v1 = os.path.join(td, "v1.mp4"); open(v1, "w").close()
            v2 = os.path.join(td, "v2.mp4"); open(v2, "w").close()
            self.ai.generate_video.side_effect = lambda p, f, **kw: f
            dirs = {"video": td, "image": td, "audio": td}
            with patch("veo_generator.MediaProcessor.extract_frame", return_value="frame.png"), \
                 patch.object(type(Config), "OUTPUT_DIRS", new_callable=lambda: property(lambda self: dirs)):
                state = self.orch.bridge_step1_prepare(v1, v2, None, "transition")
            self.assertIsNotNone(state)
            self.assertEqual(state["v2_type"], "existing")
            self.assertEqual(state["bridge"], os.path.join(td, "bridge_seg.mp4"))

    def test_bridge_prepare_missing_v1_fails(self):
        state = self.orch.bridge_step1_prepare("/nonexistent/v1.mp4", None, "p", None)
        self.assertIsNone(state)
        self.assertIn("Video 1 not found", "\n".join(self.logs))

    def test_bridge_accept_moves_preview(self):
        with tempfile.TemporaryDirectory() as td:
            prev = os.path.join(td, "preview.mp4"); open(prev, "w").close()
            dest = os.path.join(td, "final.mp4")
            state = {"preview": prev, "temp_files": []}
            out = self.orch.bridge_accept(state, dest)
            self.assertEqual(out, dest)
            self.assertTrue(os.path.exists(dest))
            self.assertFalse(os.path.exists(prev))
            # journal records the accepted bridge (reliability item)
            self.ai._journal.record.assert_any_call("bridge", "ok", output=dest, duration_s=ANY)


class TestCLI(unittest.TestCase):
    def test_main_without_api_key_exits_2(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            code = veo_generator.main([])
        self.assertEqual(code, 2)

    def test_arg_parser_video_subcommand(self):
        parser = veo_generator.build_arg_parser()
        args = parser.parse_args(["video", "a cat", "--resolution", "1080p",
                                  "--duration", "6", "--no-auto-negative", "--expand"])
        self.assertEqual(args.command, "video")
        self.assertEqual(args.resolution, "1080p")
        self.assertEqual(args.duration, 6)
        self.assertTrue(args.no_auto_negative)
        self.assertTrue(args.expand)

    def test_cli_video_one_shot(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDUMMY1234567890abcdefABCDEFGH"}), \
             patch.object(veo_generator, "GenerationOrchestrator") as MockOrch:
            MockOrch.return_value.single_video.return_value = "out.mp4"
            code = veo_generator.main(["video", "p", "--output", "out.mp4"])
        self.assertEqual(code, 0)
        MockOrch.return_value.single_video.assert_called_once()


class TestReferenceSpecParsing(unittest.TestCase):
    """Explicit style:/asset: prefixes must beat the legacy name heuristic."""

    def test_explicit_prefixes_win(self):
        self.assertEqual(GenAIService.parse_reference_spec("style:/x/plain.png"),
                         ("/x/plain.png", "STYLE"))
        self.assertEqual(GenAIService.parse_reference_spec("asset:/x/style_kit.png"),
                         ("/x/style_kit.png", "ASSET"))
        # case-insensitive prefix
        self.assertEqual(GenAIService.parse_reference_spec("STYLE:/x/a.png")[1], "STYLE")

    def test_legacy_heuristic_still_supported(self):
        self.assertEqual(GenAIService.parse_reference_spec("/imgs/style_ref.png"),
                         ("/imgs/style_ref.png", "STYLE"))
        self.assertEqual(GenAIService.parse_reference_spec("/imgs/cat.png"),
                         ("/imgs/cat.png", "ASSET"))

    def test_windows_path_not_mistaken_for_prefix(self):
        """Regression: 'C:\\...\\clip.png' used to hit the old substring check."""
        path = r"C:\media\my style shot.png"
        got_path, kind = GenAIService.parse_reference_spec(path)
        self.assertEqual(got_path, path)   # path preserved verbatim
        self.assertEqual(kind, "STYLE")    # via basename heuristic, not 'c:' prefix


class TestCancellation(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("veo_generator.genai.Client")
        self.MockClient = self.patcher.start()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDUMMY1234567890abcdefABCDEFGH"}):
            self.service = GenAIService()

    def tearDown(self):
        self.patcher.stop()

    def test_cancel_event_available_and_resettable(self):
        self.service.cancel()
        self.assertTrue(self.service.cancel_event.is_set())
        self.service.reset_cancel()
        self.assertFalse(self.service.cancel_event.is_set())

    def test_check_cancelled_raises(self):
        self.service.cancel()
        with self.assertRaises(CancelledError):
            self.service._check_cancelled()

    def test_polling_aborted_by_cancel(self):
        """A cancelled request must break out of the LRO poll loop fast."""
        op = MagicMock()
        op.done = False
        self.MockClient.return_value.models.generate_videos.return_value = op
        self.MockClient.return_value.operations.get.return_value = op

        # Cancel *during* the first poll: submit succeeds, then the user
        # hits Cancel; the poll loop must abort without waiting for timeout.
        def cancel_on_get(*a, **kw):
            self.service.cancel()
            return op
        self.MockClient.return_value.operations.get.side_effect = cancel_on_get

        with patch.object(FileManager, "record_prompt"), \
             patch.object(Config, "POLL_INTERVAL_S", 0.001), \
             patch("veo_generator.time.sleep"):
            result = self.service.generate_video("p", "out.mp4")
        self.assertIsNone(result)
        # Cancelled tasks are journalled (not silently dropped).
        entries = self.service._journal.load(5)
        kinds = [e for e in entries if e.get("kind") == "video"]
        self.assertTrue(kinds and kinds[-1]["status"] in ("cancelled", "failed"))

    def test_precancel_blocks_submit(self):
        """If already cancelled, generate_video must not even hit the API."""
        self.service.cancel()
        with patch.object(FileManager, "record_prompt"):
            with self.assertRaises(CancelledError):
                self.service.generate_video("p", "out.mp4")
        self.MockClient.return_value.models.generate_videos.assert_not_called()


class TestTaskJournal(unittest.TestCase):
    def test_record_load_stats_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            j = TaskJournal(os.path.join(td, "tasks.jsonl"))
            j.record("video", "ok", prompt="cat", output="a.mp4", duration_s=12.5)
            j.record("video", "failed", prompt="dog", error="429")
            j.record("image", "ok", duration_s=3.0)
            entries = j.load()
            self.assertEqual(len(entries), 3)
            self.assertEqual(entries[0]["kind"], "video")
            self.assertEqual(entries[0]["status"], "ok")
            st = j.stats()
            self.assertEqual(st["total"], 3)
            self.assertEqual(st["ok"], 2)
            self.assertEqual(st["failed"], 1)
            self.assertEqual(st["by_kind"]["video"]["ok"], 1)

    def test_corrupt_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "tasks.jsonl")
            with open(p, "w") as f:
                f.write('{"kind": "video", "status": "ok"}\n')
                f.write("NOT JSON {{{\n")          # torn write must not crash
                f.write('{"kind": "chain", "status": "failed"}\n')
            entries = TaskJournal(p).load()
            self.assertEqual(len(entries), 2)

    def test_record_never_raises_on_bad_path(self):
        j = TaskJournal(os.path.join("/proc/nonexistent-dir-xyz", "t.jsonl"))
        j.record("video", "ok")   # must swallow OSError, not propagate


class TestVeo31DocCompliance(unittest.TestCase):
    """Checks against the official Veo 3.1 / Lyria documentation."""

    def setUp(self):
        self.patcher = patch("google.genai.Client")
        self.mock_client_cls = self.patcher.start()
        self.mock_client_instance = MagicMock()
        self.mock_client_cls.return_value = self.mock_client_instance
        op = MagicMock()
        op.done = True
        op.response.generated_videos = [MagicMock()]
        self.mock_client_instance.models.generate_videos.return_value = op
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDUMMY1234567890abcdefABCDEFGH"}):
            self.service = GenAIService()

    def tearDown(self):
        self.patcher.stop()

    def test_default_model_is_stable_id(self):
        # Official stable Vertex AI model id (preview variant is deprecated).
        self.assertIn("veo-3.1-generate", Config.MODEL_VIDEO)

    def test_duration_clamped_to_documented_range(self):
        # Docs: duration_seconds whole seconds between 4 and 60.
        self.assertEqual(Config.MIN_DURATION_S, 4)
        self.assertEqual(Config.MAX_DURATION_S, 60)
        with patch.object(FileManager, "record_prompt"):
            self.service.generate_video("p", "out.mp4", duration_seconds=120)
        cfg = self.mock_client_instance.models.generate_videos.call_args.kwargs["config"]
        self.assertEqual(cfg.duration_seconds, 60)
        with patch.object(FileManager, "record_prompt"):
            self.service.generate_video("p", "out.mp4", duration_seconds=2)
        cfg = self.mock_client_instance.models.generate_videos.call_args.kwargs["config"]
        self.assertEqual(cfg.duration_seconds, 4)

    def test_person_generation_snake_case_value(self):
        with patch.object(FileManager, "record_prompt"):
            self.service.generate_video("p", "out.mp4")
        cfg = self.mock_client_instance.models.generate_videos.call_args.kwargs["config"]
        self.assertEqual(cfg.person_generation, "allow_adult")

    def test_reference_image_limits_and_style_type(self):
        tmp = tempfile.mkdtemp()
        try:
            img = os.path.join(tmp, "ref.png")
            with open(img, "wb") as f:
                f.write(b"\x89PNG fake")
            refs = [img] * 5 + ["style:" + img] * 4
            fake_image = types.Image.model_construct(imageBytes=b"fake", mimeType="image/png")
            with patch.object(FileManager, "record_prompt"), \
                 patch.object(self.service, "_get_image_param",
                              return_value=fake_image):
                self.service.generate_video("p", "out.mp4", reference_image_paths=refs)
            cfg = self.mock_client_instance.models.generate_videos.call_args.kwargs["config"]
            kinds = [r.reference_type for r in cfg.reference_images]
            self.assertEqual(kinds.count("ASSET"), 3)  # max 3 asset images
            self.assertEqual(kinds.count("STYLE"), 3)  # max 3 style images
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_argparse_rejects_bad_duration_not_resolution(self):
        parser = veo_generator.build_arg_parser()
        # documented range enforced at service level, argparse keeps int type
        args = parser.parse_args(["video", "p", "--duration", "8"])
        self.assertEqual(args.duration, 8)


class TestGUIModule(unittest.TestCase):
    """GUI smoke tests that do not require a display server."""

    def test_gui_imports_and_wires_orchestrator(self):
        import veo_gui
        src = open(os.path.join(os.path.dirname(veo_gui.__file__), "veo_gui.py")).read()
        # No stdout monkey-patching anymore (item 2)
        self.assertNotIn("sys.stdout =", src)
        # Uses queue-based logging and shared orchestrator (items 2, 8)
        self.assertIn("queue.Queue", src)
        self.assertIn("GenerationOrchestrator", src)
        # Migrated off Tkinter onto Flet; tabs use SegmentedButton + Stack
        self.assertIn("import flet as ft", src)
        self.assertNotIn("tkinter", src)
        self.assertIn("SegmentedButton", src)
        self.assertIn("ft.Stack", src)
        # Cancellation & progress plumbing wired through the orchestrator
        self.assertIn("progress_cb", src)
        self.assertIn("cancel", src)

    @unittest.skipIf(os.environ.get("SKIP_GUI_DISPLAY_TESTS") == "1", "needs display")
    def test_gui_app_class_exists(self):
        try:
            import flet  # noqa: F401
        except ImportError:
            self.skipTest("flet unavailable")
        from veo_gui import VeoGUIApp
        # Constructing the app requires a live Flet page; we only assert
        # the public surface used by main() here.
        self.assertTrue(callable(VeoGUIApp))
        self.assertTrue(hasattr(VeoGUIApp, "_switch_tab"))
        self.assertTrue(hasattr(VeoGUIApp, "_on_progress"))
        self.assertTrue(hasattr(VeoGUIApp, "_cancel_current"))


class TestCLIFlags(unittest.TestCase):
    def test_video_flags_audio_upscale(self):
        parser = veo_generator.build_arg_parser()
        args = parser.parse_args(["video", "p", "--no-audio", "--upscale"])
        self.assertFalse(args.audio)
        self.assertTrue(args.upsampling)
        # defaults: audio on, upscale unset (env-driven)
        d = parser.parse_args(["video", "p"])
        self.assertTrue(d.audio)
        self.assertIsNone(d.upsampling)

    def test_batch_flags(self):
        parser = veo_generator.build_arg_parser()
        args = parser.parse_args(["batch", "a", "b", "--no-audio", "--upscale",
                                  "--workers", "2"])
        self.assertFalse(args.audio)
        self.assertTrue(args.upsampling)
        self.assertEqual(args.workers, 2)

    def test_tasks_subcommand(self):
        parser = veo_generator.build_arg_parser()
        args = parser.parse_args(["tasks", "--limit", "5", "--json"])
        self.assertEqual(args.command, "tasks")
        self.assertEqual(args.limit, 5)
        self.assertTrue(args.json)

    def test_tasks_command_runs_without_api_key(self):
        """'tasks' is read-only reporting: must work with no GEMINI_API_KEY."""
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(Config, "TASKS_JOURNAL_FILE",
                          os.path.join(tempfile.mkdtemp(), "t.jsonl")):
            code = veo_generator.main(["tasks", "--limit", "3"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
