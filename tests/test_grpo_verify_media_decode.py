"""Focused dependency-light tests for the mixed-media decode preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

VERIFIER = SRC / "grpo_verify_media_decode.py"
spec = importlib.util.spec_from_file_location("media_decode_verifier_under_test", VERIFIER)
assert spec is not None and spec.loader is not None
media_decode_verifier = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = media_decode_verifier
spec.loader.exec_module(media_decode_verifier)


class MixedMediaVerifierConfigurationTests(unittest.TestCase):
    def test_nframes_matches_qwen_even_frame_contract(self) -> None:
        self.assertEqual(media_decode_verifier._even_nframes("8"), 8)
        for value in ("0", "1", "9"):
            with self.assertRaises(Exception):
                media_decode_verifier._even_nframes(value)

    def test_media_element_mirrors_current_trainer_modality_settings(self) -> None:
        path = Path("/safe/root/item")
        image = media_decode_verifier._build_media_element(
            "image", path, nframes=8, max_pixels=401408, image_max_pixels=None
        )
        video = media_decode_verifier._build_media_element(
            "video", path, nframes=8, max_pixels=401408, image_max_pixels=None
        )

        self.assertEqual(image, {"type": "image", "image": str(path)})
        self.assertEqual(
            video,
            {
                "type": "video",
                "video": str(path),
                "nframes": 8,
                "max_pixels": 401408,
            },
        )

    def test_optional_image_cap_is_explicit(self) -> None:
        image = media_decode_verifier._build_media_element(
            "image",
            Path("/safe/root/image.jpg"),
            nframes=8,
            max_pixels=401408,
            image_max_pixels=401408,
        )
        self.assertEqual(image["max_pixels"], 401408)

    def test_modality_counts_keep_unsupported_records_visible(self) -> None:
        records = [
            {"data_type": "image"},
            {"data_type": "video"},
            {"data_type": "audio"},
            {},
        ]
        self.assertEqual(
            media_decode_verifier._count_records(records),
            {"image": 1, "video": 1, "unsupported": 2},
        )
        self.assertEqual(
            media_decode_verifier._count_records(records, {1, 3}),
            {"image": 0, "video": 1, "unsupported": 1},
        )

    def test_safe_path_helper_rejects_parent_escape_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = root / "safe.jpg"
            safe.write_bytes(b"not-a-real-image")
            self.assertEqual(
                media_decode_verifier._safe_regular_nonempty_path(root, "./safe.jpg"),
                safe.resolve(),
            )
            self.assertIsNone(
                media_decode_verifier._safe_regular_nonempty_path(root, "../safe.jpg")
            )
            self.assertIsNone(
                media_decode_verifier._safe_regular_nonempty_path(root, "/safe.jpg")
            )


if __name__ == "__main__":
    unittest.main()
