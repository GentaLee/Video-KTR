"""Verify cache collation ordering, safety, numeric parity, and CPU spawning."""

from __future__ import annotations

import copy
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grpo_media_prefetch import (  # noqa: E402
    CachedMediaCollator,
    MEDIA_PAYLOAD_KEY,
    PreparedMedia,
    use_spawn_for_cpu_prefetch,
)


def add_trainer_import_paths() -> None:
    for candidate in reversed((
        ROOT / ".python-packages-grpo",
        ROOT / "src" / "r1-v" / "src",
        ROOT / "src" / "qwen-vl-utils" / "src",
    )):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


class CachedMediaPrefetchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.media = self.root / "media"
        self.media.mkdir()
        for name in ("image.png", "video.mp4"):
            (self.media / name).write_bytes(b"media existence checked separately from cache")
        self.image = Image.new("RGB", (4, 5), (10, 20, 30))
        self.video = torch.arange(8 * 3 * 4 * 5, dtype=torch.float32).reshape(8, 3, 4, 5)
        self.calls: list[tuple[dict, dict]] = []

        def load_cached_media(element: dict, **kwargs):
            self.calls.append((element, kwargs))
            if element["type"] == "image":
                return self.image.copy(), None
            return self.video.clone(), 0.625

        self.cache = ModuleType("grpo_media_cache")
        self.cache.load_cached_media = load_cached_media
        self.collator = CachedMediaCollator(str(self.media), str(self.root / "cache"), 8, 401408)

    def features(self) -> list[dict]:
        return [
            {"problem_id": 2, "data_type": "video", "path": "./video.mp4", "prompt": [{"role": "user"}]},
            {"problem_id": 1, "data_type": "image", "path": "image.png", "prompt": [{"role": "user"}]},
            {"problem_id": 3, "data_type": "video", "path": "video.mp4", "prompt": [{"role": "user"}]},
        ]

    def test_preserves_records_order_values_cpu_payload_and_rng(self) -> None:
        features = self.features()
        snapshot = copy.deepcopy(features)
        torch_rng = torch.random.get_rng_state()
        python_rng = random.getstate()
        with patch.dict(sys.modules, {"grpo_media_cache": self.cache}):
            result = self.collator(features)
        self.assertEqual(features, snapshot)
        self.assertEqual([r["problem_id"] for r in result], [2, 1, 3])
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        self.assertIsNot(result[0], features[0])
        video = result[0][MEDIA_PAYLOAD_KEY]
        self.assertIsInstance(video, PreparedMedia)
        self.assertTrue(torch.equal(video.video_inputs[0], self.video))
        self.assertEqual(video.video_inputs[0].device.type, "cpu")
        self.assertEqual(video.video_kwargs, {"fps": [0.625]})
        self.assertEqual(result[1][MEDIA_PAYLOAD_KEY].image_inputs[0].tobytes(), self.image.tobytes())
        self.assertEqual(result[1][MEDIA_PAYLOAD_KEY].video_kwargs, {"fps": []})
        self.assertTrue(all(kwargs["allow_create"] is False for _, kwargs in self.calls))
        self.assertEqual(self.calls[0][0]["nframes"], 8)
        self.assertEqual(self.calls[0][0]["max_pixels"], 401408)
        self.assertNotIn("max_pixels", self.calls[1][0])
        roundtrip = pickle.loads(pickle.dumps(result))
        self.assertTrue(torch.equal(roundtrip[0][MEDIA_PAYLOAD_KEY].video_inputs[0], self.video))
        pickle.loads(pickle.dumps(self.collator))

    def test_rejects_unsafe_missing_empty_and_symlink_escape_before_cache(self) -> None:
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        (self.media / "escape.png").symlink_to(outside)
        (self.media / "empty.png").touch()
        for path in ("../outside.png", str(outside), "escape.png", "empty.png", "missing.png"):
            with self.subTest(path=path), patch.dict(sys.modules, {"grpo_media_cache": self.cache}):
                with self.assertRaises(ValueError):
                    self.collator([{"data_type": "image", "path": path}])
        self.assertEqual(self.calls, [])

    def test_cache_miss_does_not_decode_or_filter_sample(self) -> None:
        def fail(*args, **kwargs):
            raise RuntimeError("cache entry is stale or missing")

        self.cache.load_cached_media = fail
        with patch.dict(sys.modules, {"grpo_media_cache": self.cache}):
            with self.assertRaisesRegex(RuntimeError, "stale or missing"):
                self.collator(self.features())

    def test_rejects_requantized_video_and_reserved_field(self) -> None:
        self.cache.load_cached_media = lambda *a, **k: (self.video.to(torch.uint8), 0.625)
        with patch.dict(sys.modules, {"grpo_media_cache": self.cache}):
            with self.assertRaisesRegex(ValueError, "float32"):
                self.collator(self.features()[:1])
            with self.assertRaisesRegex(ValueError, "reserved"):
                self.collator([{**self.features()[0], MEDIA_PAYLOAD_KEY: None}])

    def test_zero_worker_loader_keeps_original_sampler_and_main_threads(self) -> None:
        loader = DataLoader(self.features(), batch_size=1, collate_fn=self.collator, num_workers=0)
        sampler = loader.sampler
        self.assertIs(use_spawn_for_cpu_prefetch(loader), loader)
        self.assertIs(loader.sampler, sampler)
        self.assertIsNone(loader.multiprocessing_context)
        with patch.dict(sys.modules, {"grpo_media_cache": self.cache}), patch("torch.set_num_threads") as threads:
            self.assertEqual([batch[0]["problem_id"] for batch in loader], [2, 1, 3])
            threads.assert_not_called()

    def test_accelerate_adapter_targets_base_loader_and_opaque_payload(self) -> None:
        loader = DataLoader(self.features(), batch_size=1, collate_fn=self.collator, num_workers=2)
        wrapper = SimpleNamespace(base_dataloader=loader)
        sampler = loader.sampler
        self.assertIs(use_spawn_for_cpu_prefetch(wrapper), wrapper)
        self.assertIs(loader.sampler, sampler)
        self.assertEqual(loader.multiprocessing_context.get_start_method(), "spawn")
        # Accelerate's documented send_to_device only recurses dict/tuple/list
        # and objects exposing .to(); the opaque wrapper must expose none.
        with patch.dict(sys.modules, {"grpo_media_cache": self.cache}):
            payload = self.collator(self.features()[:1])[0][MEDIA_PAYLOAD_KEY]
        self.assertFalse(isinstance(payload, (dict, tuple, list)))
        self.assertFalse(hasattr(payload, "to"))

    def test_real_spawn_workers_preserve_batch_order(self) -> None:
        # Put a deterministic read-only cache fixture on the inherited import
        # path; unlike an in-memory mock it is importable by spawned workers.
        fixture = self.root / "fixture"
        fixture.mkdir()
        (fixture / "grpo_media_cache.py").write_text(
            'import torch\nfrom PIL import Image\n'
            'def load_cached_media(element, *, cache_root, allow_create=False):\n'
            '    assert allow_create is False\n'
            '    assert torch.get_num_threads() == 1\n'
            '    if element["type"] == "image":\n'
            '        return Image.new("RGB", (4, 5), (10, 20, 30)), None\n'
            '    return torch.arange(480, dtype=torch.float32).reshape(8, 3, 4, 5), 0.625\n',
            encoding="utf-8",
        )
        with patch.object(sys, "path", [str(fixture), *sys.path]):
            loader = DataLoader(
                self.features(), batch_size=1, collate_fn=self.collator,
                num_workers=2, prefetch_factor=2, persistent_workers=False,
            )
            use_spawn_for_cpu_prefetch(loader)
            batches = list(loader)
        self.assertEqual([batch[0]["problem_id"] for batch in batches], [2, 1, 3])
        self.assertTrue(torch.equal(batches[0][0][MEDIA_PAYLOAD_KEY].video_inputs[0], self.video))

    def test_real_accelerate_sharding_and_device_movement_preserve_payload(self) -> None:
        add_trainer_import_paths()
        from accelerate.data_loader import prepare_data_loader
        from accelerate.utils import send_to_device

        records = self.features() + [{**self.features()[1], "problem_id": 4}]
        with patch.dict(sys.modules, {"grpo_media_cache": self.cache}):
            for rank, expected in ((0, [2, 3]), (1, [1, 4])):
                loader = DataLoader(records, batch_size=1, collate_fn=self.collator, num_workers=0)
                shard = prepare_data_loader(loader, num_processes=2, process_index=rank)
                use_spawn_for_cpu_prefetch(shard)
                self.assertEqual([batch[0]["problem_id"] for batch in shard], expected)
            result = self.collator(records[:1])
        payload = result[0][MEDIA_PAYLOAD_KEY]
        moved = send_to_device(result, torch.device("meta"))
        self.assertIs(moved[0][MEDIA_PAYLOAD_KEY], payload)
        self.assertEqual(moved[0][MEDIA_PAYLOAD_KEY].video_inputs[0].device.type, "cpu")

    def test_trainer_cache_and_uncached_processor_inputs_match(self) -> None:
        add_trainer_import_paths()
        from open_r1.trainer import video_ktr_grpo_trainer as module

        class Processor:
            def __call__(inner_self, **kwargs):
                captured.append(kwargs)
                value = kwargs["videos"][0] if kwargs["videos"] else torch.tensor(list(kwargs["images"][0].getdata()))
                return {"input_ids": torch.tensor([[1, 2, 3]]), "pixels_for_test": value.clone()}

        trainer = object.__new__(module.VideoKTRGRPOTrainer)
        trainer.data_root = self.media
        trainer.nframes = 8
        trainer.max_pixels = 401408
        trainer.max_prompt_length = 16384
        trainer.processing_class = Processor()
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer._rank_trace = lambda *args, **kwargs: None
        for modality, path in (("video", "video.mp4"), ("image", "image.png")):
            feature = {"data_type": modality, "path": path, "prompt": [{"role": "user", "content": [{"type": modality}]}]}
            media_result = (None, [self.video], {"fps": [0.625]}) if modality == "video" else ([self.image], None, {"fps": []})
            captured = []
            with patch.object(module, "maybe_apply_chat_template", return_value={"prompt": "fixed prompt"}), patch.object(module, "process_vision_info", return_value=media_result) as decode:
                trainer._media_cache_root = ""
                ordinary = trainer._prepare_prompt([feature])
                trainer._media_cache_root = self.collator.cache_root
                with patch.dict(sys.modules, {"grpo_media_cache": self.cache}):
                    features = self.collator([feature])
                cached = trainer._prepare_prompt(features)
                self.assertEqual(decode.call_count, 1)
                self.assertEqual(ordinary[0], cached[0])
                self.assertEqual(ordinary[3], cached[3])
                self.assertEqual(ordinary[5], cached[5])
                for key in ordinary[4]:
                    self.assertTrue(torch.equal(ordinary[4][key], cached[4][key]))
                features[0][MEDIA_PAYLOAD_KEY].max_pixels = 1234
                with self.assertRaisesRegex(RuntimeError, "settings changed"):
                    trainer._prepare_prompt(features)


if __name__ == "__main__":
    unittest.main()
