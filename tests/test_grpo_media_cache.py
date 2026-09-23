"""Exact-cache invariance, invalidation and failure tests (no model/GPU)."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grpo_media_cache as cache

try:
    import torch
    from PIL import Image
except ImportError:
    torch = Image = None


class RuntimeFingerprintTests(unittest.TestCase):
    def test_qwen_source_and_environment_are_fingerprinted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "vision_process.py"
            source.write_text("# first decoder revision\n")
            spec = SimpleNamespace(submodule_search_locations=[str(root)])
            with patch.object(cache.importlib.util, "find_spec", return_value=spec), \
                 patch.object(cache.importlib.metadata, "version", return_value="1.0"), \
                 patch.dict(os.environ, {"VIDEO_MAX_PIXELS": "123", "FORCE_QWENVL_VIDEO_READER": "torchvision"}):
                first = cache.runtime_fingerprint()
                self.assertEqual(first["environment"]["VIDEO_MAX_PIXELS"], "123")
                self.assertEqual(first["versions"]["torch"], "1.0")
                self.assertEqual(first["qwen_vision_process_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
                source.write_text("# changed decoder revision\n")
                self.assertNotEqual(first, cache.runtime_fingerprint())


@unittest.skipIf(torch is None or Image is None, "requires torch and Pillow")
class ExactMediaCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.mp4"
        self.source.write_bytes(b"test-source-media")
        self.element = {"type": "video", "video": str(self.source),
                        "nframes": 8, "max_pixels": 401408}
        self.cache_root = self.root / "persistent-cache"
        self.fingerprint = {"versions": {"torch": "test"}, "environment": {},
                            "qwen_vision_process_sha256": "a" * 64}
        self.runtime = patch.object(cache, "runtime_fingerprint", return_value=self.fingerprint)
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        # Non-integral/negative values catch accidental uint8 image conversion.
        self.video = torch.linspace(-2.75, 258.25, 8 * 3 * 4 * 6).reshape(8, 3, 4, 6)
        self.decoder = patch.object(cache, "_decode", return_value=(self.video, 1.37)).start()
        self.addCleanup(patch.stopall)

    def load(self, **kwargs):
        return cache.load_cached_media(self.element, cache_root=self.cache_root, **kwargs)

    def metadata_path(self):
        return next(self.cache_root.rglob("metadata.json"))

    def test_video_bitwise_equal_and_fps_preserved_on_hit(self):
        created, fps = self.load(allow_create=True)
        hit, hit_fps = self.load()
        self.assertTrue(torch.equal(created, self.video))
        self.assertTrue(torch.equal(hit, self.video))
        self.assertEqual(hit.dtype, torch.float32)
        self.assertEqual(hit.device.type, "cpu")
        self.assertEqual((fps, hit_fps), (1.37, 1.37))
        self.assertEqual(self.decoder.call_count, 1)
        metadata = json.loads(self.metadata_path().read_text())
        self.assertEqual(metadata["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        hit.zero_()
        self.assertTrue(torch.equal(self.load()[0], self.video))

    def test_lossless_rgb_image_pixels(self):
        original = Image.new("RGB", (7, 5), (19, 37, 243))
        original.putpixel((3, 2), (255, 0, 9))
        self.element = {"type": "image", "image": str(self.source)}
        self.decoder.return_value = (original, None)
        created, fps = self.load(allow_create=True)
        loaded, _ = self.load()
        self.assertEqual(created.tobytes(), original.tobytes())
        self.assertEqual(loaded.tobytes(), original.tobytes())
        self.assertEqual(loaded.mode, "RGB")
        self.assertIsNone(fps)

    def test_missing_cache_never_decodes_or_creates_directories(self):
        with self.assertRaisesRegex(cache.MediaCacheError, "missing or invalidated"):
            self.load()
        self.decoder.assert_not_called()
        self.assertFalse(self.cache_root.exists())

    def test_source_stat_change_invalidates_cache(self):
        self.load(allow_create=True)
        self.source.write_bytes(b"updated-source-media")
        with self.assertRaisesRegex(cache.MediaCacheError, "missing or invalidated"):
            self.load()
        self.assertEqual(self.decoder.call_count, 1)

    def test_ctime_detects_same_size_rewrite_with_restored_mtime(self):
        self.load(allow_create=True)
        info = self.source.stat()
        before = cache._source_identity(self.source)
        self.assertEqual(before["ctime_ns"], info.st_ctime_ns)
        self.source.write_bytes(b"TEST-SOURCE-MEDIA")
        os.utime(self.source, ns=(info.st_atime_ns, info.st_mtime_ns))
        after = self.source.stat()
        self.assertEqual(after.st_size, info.st_size)
        self.assertEqual(after.st_mtime_ns, info.st_mtime_ns)
        # A fast rewrite can occur within one filesystem timestamp tick.
        # Isolate ctime as the ONLY identity change rather than depending
        # on wall time or the filesystem's effective timestamp precision.
        changed = {**before, "ctime_ns": before["ctime_ns"] + 1}
        with patch.object(cache, "_source_identity", return_value=changed):
            with self.assertRaisesRegex(cache.MediaCacheError, "missing or invalidated"):
                self.load()
        self.assertEqual(self.decoder.call_count, 1)

    def test_all_preprocessing_parameters_invalidate(self):
        self.load(allow_create=True)
        for name, value in (("nframes", 16), ("max_pixels", 999),
                            ("video_start", 2.0), ("total_pixels", 12345)):
            with self.subTest(name=name):
                changed = {**self.element, name: value}
                with self.assertRaisesRegex(cache.MediaCacheError, "missing or invalidated"):
                    cache.load_cached_media(changed, cache_root=self.cache_root)

    def test_runtime_or_environment_change_invalidates(self):
        self.load(allow_create=True)
        for fingerprint in (
            {**self.fingerprint, "versions": {"torch": "different"}},
            {**self.fingerprint, "environment": {"VIDEO_MAX_PIXELS": "42"}},
            {**self.fingerprint, "qwen_vision_process_sha256": "b" * 64},
        ):
            with patch.object(cache, "runtime_fingerprint", return_value=fingerprint):
                with self.assertRaisesRegex(cache.MediaCacheError, "missing or invalidated"):
                    self.load()

    def test_payload_corruption_refused_without_decode_fallback(self):
        self.load(allow_create=True)
        (self.metadata_path().parent / "video.pt").write_bytes(b"corrupt")
        for allow_create in (False, True):
            with self.assertRaisesRegex(cache.MediaCacheError, "SHA256 mismatch"):
                self.load(allow_create=allow_create)
        self.assertEqual(self.decoder.call_count, 1)

    def test_incomplete_certificate_refused(self):
        self.load(allow_create=True)
        metadata_path = self.metadata_path()
        metadata = json.loads(metadata_path.read_text())
        metadata["status"] = "pending"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(cache.MediaCacheError, "certificate"):
            self.load()

    def test_decode_failure_publishes_no_entry(self):
        self.decoder.side_effect = RuntimeError("decoder failure")
        with self.assertRaisesRegex(cache.MediaCacheError, "decoder failure"):
            self.load(allow_create=True)
        self.assertFalse(list(self.cache_root.rglob("metadata.json")))
        self.assertFalse(list(self.cache_root.rglob("*.pending-*")))

    def test_source_modified_during_decode_publishes_no_entry(self):
        def mutate_source(_):
            self.source.write_bytes(b"changed while decoding")
            return self.video, 1.37
        self.decoder.side_effect = mutate_source
        with self.assertRaisesRegex(cache.MediaCacheError, "changed during cache"):
            self.load(allow_create=True)
        self.assertFalse(list(self.cache_root.rglob("metadata.json")))

    def test_float16_decoder_output_refused_instead_of_converted(self):
        self.decoder.return_value = (self.video.half(), 1.37)
        with self.assertRaisesRegex(cache.MediaCacheError, "CPU float32"):
            self.load(allow_create=True)
        self.assertFalse(list(self.cache_root.rglob("metadata.json")))

    def test_concurrent_creators_only_decode_once(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.load(allow_create=True), range(4)))
        self.assertEqual(self.decoder.call_count, 1)
        self.assertTrue(all(torch.equal(result[0], self.video) for result in results))
        self.assertEqual(len(list(self.cache_root.rglob("metadata.json"))), 1)


if __name__ == "__main__":
    unittest.main()
