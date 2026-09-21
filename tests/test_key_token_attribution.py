"""Unit tests for the standalone Video-KTR token attribution utilities.

This file deliberately uses ``unittest`` so it can run directly even where
pytest is not installed:

    python tests/test_key_token_attribution.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from key_token_attribution import (  # noqa: E402
    attribute_key_tokens,
    compute_logp_delta,
    compute_token_entropy,
    keep_token_by_ratio,
    select_top_ratio,
    target_token_logps,
    union_masks,
)
from grpo_prepare_dataset import resolve_media_path  # noqa: E402


PACKAGE_HELPER = ROOT / "src" / "r1-v" / "src" / "open_r1" / "trainer" / "ktr_token_utils.py"
_helper_spec = importlib.util.spec_from_file_location("ktr_token_utils_under_test", PACKAGE_HELPER)
assert _helper_spec is not None and _helper_spec.loader is not None
ktr_token_utils = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(ktr_token_utils)


class TokenEntropyTests(unittest.TestCase):
    def test_entropy_matches_uniform_and_near_deterministic_distributions(self) -> None:
        logits = torch.tensor([[[0.0, 0.0], [20.0, -20.0]]])

        entropy = compute_token_entropy(logits)

        self.assertTrue(torch.allclose(entropy[0, 0], torch.log(torch.tensor(2.0))))
        self.assertLess(entropy[0, 1].item(), 1e-12)

    def test_target_logps_extracts_teacher_forced_targets(self) -> None:
        logits = torch.tensor([[[0.0, 0.0], [0.0, torch.log(torch.tensor(3.0))]]])
        target_ids = torch.tensor([[0, 1]], dtype=torch.long)

        actual = target_token_logps(logits, target_ids)
        expected = torch.tensor([[-torch.log(torch.tensor(2.0)), torch.log(torch.tensor(0.75))]])

        self.assertTrue(torch.allclose(actual, expected))


class DeltaTests(unittest.TestCase):
    def test_absolute_and_signed_deltas_have_explicit_semantics(self) -> None:
        original = torch.tensor([[-1.0, -4.0]])
        perturbed = torch.tensor([[-2.0, -1.0]])

        signed = compute_logp_delta(original, perturbed, mode="signed")
        absolute = compute_logp_delta(original, perturbed, mode="absolute")

        self.assertTrue(torch.equal(signed, torch.tensor([[1.0, -3.0]])))
        self.assertTrue(torch.equal(absolute, torch.tensor([[1.0, 3.0]])))
        self.assertTrue(torch.equal(compute_logp_delta(original, perturbed, mode="repo"), signed))
        self.assertTrue(torch.equal(compute_logp_delta(original, perturbed, mode="paper"), absolute))


class PackagedTrainerHelperParityTests(unittest.TestCase):
    """Keep the package-local trainer helper aligned with standalone evidence code."""

    def test_tensor_scoring_and_selection_match_standalone_utilities(self) -> None:
        logits = torch.tensor(
            [
                [[0.1, 0.3, -0.2], [0.4, -0.1, 0.0], [0.2, 0.2, 0.2]],
                [[-0.4, 0.5, 0.1], [0.0, 0.6, -0.3], [0.8, -0.2, 0.1]],
            ]
        )
        targets = torch.tensor([[1, 0, 2], [1, 2, 0]], dtype=torch.long)
        original = target_token_logps(logits, targets)
        perturbed = original + torch.tensor([[0.2, -0.1, 0.4], [-0.3, 0.5, -0.2]])
        valid = torch.tensor([[True, True, False], [True, True, True]])

        self.assertTrue(torch.allclose(ktr_token_utils.compute_token_entropy(logits), compute_token_entropy(logits)))
        self.assertTrue(torch.allclose(ktr_token_utils.target_token_logps(logits, targets), original))
        for mode, scope in (("paper", "per_completion"), ("repo", "batch")):
            expected_delta = compute_logp_delta(original, perturbed, mode=mode)
            actual_delta = ktr_token_utils.compute_logp_delta(original, perturbed, mode=mode)
            self.assertTrue(torch.equal(actual_delta, expected_delta))
            expected_mask = select_top_ratio(expected_delta, valid, 0.5, mode=mode, scope=scope)
            actual_mask = ktr_token_utils.select_top_ratio(actual_delta, valid, 0.5, mode=mode, scope=scope)
            self.assertTrue(torch.equal(actual_mask, expected_mask))
            self.assertTrue(
                torch.equal(
                    ktr_token_utils.union_masks(actual_mask, valid, valid_mask=valid),
                    union_masks(expected_mask, valid, valid_mask=valid),
                )
            )

    def test_completion_mask_and_temporal_permutations_match_contract(self) -> None:
        completion_ids = torch.tensor([[5, 9, 0, 0], [2, 3, 4, 5]])
        expected_mask = torch.tensor([[True, False, False, False], [True, True, True, True]])
        self.assertTrue(
            torch.equal(ktr_token_utils.completion_valid_mask(completion_ids, 9, 0), expected_mask)
        )
        permutations = ktr_token_utils.nonidentity_frame_permutations(
            4, 3, 17, include_reverse=True
        )
        self.assertEqual(permutations[0], [3, 2, 1, 0])
        self.assertEqual(len(permutations), 3)
        self.assertEqual(len({tuple(item) for item in permutations}), 3)
        self.assertNotIn([0, 1, 2, 3], permutations)


class DatasetPathSafetyTests(unittest.TestCase):
    def test_prepare_dataset_rejects_parent_traversal_before_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = root / "safe.mp4"
            safe.write_bytes(b"video")
            self.assertEqual(resolve_media_path(root, "./safe.mp4"), safe.resolve())
            self.assertIsNone(resolve_media_path(root, "../safe.mp4"))
            self.assertIsNone(resolve_media_path(root, "nested/../safe.mp4"))
            self.assertIsNone(resolve_media_path(root, "/safe.mp4"))


class SelectionTests(unittest.TestCase):
    def test_repo_batch_quantile_keeps_boundary_ties(self) -> None:
        scores = torch.tensor([[5.0, 4.0, 4.0, 1.0]])
        valid = torch.ones_like(scores, dtype=torch.bool)

        selected = select_top_ratio(scores, valid, 0.5, mode="repo")

        # Median threshold is 4, so the repository rule retains all three
        # tokens at or above it rather than forcing an exact count of two.
        self.assertTrue(torch.equal(selected, torch.tensor([[True, True, True, False]])))
        self.assertTrue(torch.equal(keep_token_by_ratio(scores, valid, 0.5), selected))

    def test_paper_per_completion_selects_exact_topk_and_excludes_invalid(self) -> None:
        scores = torch.tensor(
            [
                [100.0, 5.0, 4.0, 4.0, 1.0],
                [1.0, 1.0, 0.0, -1.0, -2.0],
            ]
        )
        valid = torch.tensor(
            [
                [False, True, True, True, True],
                [True, True, True, False, False],
            ]
        )

        selected = select_top_ratio(scores, valid, 0.5, mode="paper")

        # Four valid tokens in row 0 -> two selected. Three in row 1 ->
        # ceil(1.5) = two. Stable tie handling chooses the earlier index.
        expected = torch.tensor(
            [
                [False, True, True, False, False],
                [True, True, False, False, False],
            ]
        )
        self.assertTrue(torch.equal(selected, expected))
        self.assertTrue(torch.equal(selected.sum(dim=1), torch.tensor([2, 2])))

    def test_zero_ratio_and_empty_completion_select_nothing(self) -> None:
        scores = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        valid = torch.tensor([[True, True], [False, False]])

        self.assertFalse(select_top_ratio(scores, valid, 0.0, mode="paper").any().item())
        self.assertFalse(select_top_ratio(scores, valid, 0.5, mode="paper")[1].any().item())

    def test_invalid_inputs_fail_fast(self) -> None:
        scores = torch.tensor([[1.0, float("nan")]])
        valid = torch.tensor([[True, True]])
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            select_top_ratio(scores, valid, 0.5)
        with self.assertRaisesRegex(ValueError, "ratio"):
            select_top_ratio(torch.ones(1, 2), valid, 1.1)
        with self.assertRaisesRegex(ValueError, "shape"):
            select_top_ratio(torch.ones(1, 2), torch.ones(2, dtype=torch.bool), 0.5)


class UnionAndIntegrationTests(unittest.TestCase):
    def test_union_is_boolean_and_restricted_to_valid_positions(self) -> None:
        entropy = torch.tensor([[True, False, False, True]])
        visual = torch.tensor([[False, True, False, True]])
        temporal = torch.tensor([[False, False, True, False]])
        valid = torch.tensor([[True, True, True, False]])

        union = union_masks(entropy, visual, temporal, valid_mask=valid)

        self.assertTrue(torch.equal(union, torch.tensor([[True, True, True, False]])))
        self.assertEqual(union.dtype, torch.bool)

    def test_integrated_paper_attribution_has_three_masks_and_union(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0],  # high entropy; stable tie wins position 0
                    [20.0, -20.0],
                    [0.0, 0.0],
                    [3.0, 0.0],  # invalid/padded completion position
                ]
            ]
        )
        original = torch.zeros(1, 4)
        visual = torch.tensor([[-0.1, -4.0, -0.2, -9.0]])
        temporal = torch.tensor([[-0.1, -0.2, -5.0, -9.0]])
        valid = torch.tensor([[True, True, True, False]])

        result = attribute_key_tokens(
            logits,
            original,
            visual,
            valid,
            temporal_logps=temporal,
            entropy_ratio=0.3,
            visual_ratio=0.3,
            temporal_ratio=0.3,
            mode="paper",
        )

        self.assertTrue(torch.equal(result.entropy_mask, torch.tensor([[True, False, False, False]])))
        self.assertTrue(torch.equal(result.visual_mask, torch.tensor([[False, True, False, False]])))
        self.assertTrue(torch.equal(result.temporal_mask, torch.tensor([[False, False, True, False]])))
        self.assertTrue(torch.equal(result.union_mask, torch.tensor([[True, True, True, False]])))
        self.assertEqual(result.mode, "paper")
        self.assertEqual(result.scope, "per_completion")

    def test_integrated_attribution_accepts_preaggregated_temporal_scores(self) -> None:
        logits = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]])
        original = torch.zeros(1, 3)
        visual = torch.tensor([[-1.0, -2.0, -3.0]])
        temporal_scores = torch.tensor([[0.2, 0.9, 0.1]])
        valid = torch.ones(1, 3, dtype=torch.bool)

        result = attribute_key_tokens(
            logits,
            original,
            visual,
            valid,
            temporal_delta_scores=temporal_scores,
            entropy_ratio=0.33,
            visual_ratio=0.33,
            temporal_ratio=0.33,
            mode="paper",
        )

        self.assertTrue(torch.equal(result.temporal_delta, temporal_scores))
        self.assertTrue(torch.equal(result.temporal_mask, torch.tensor([[False, True, False]])))
        with self.assertRaisesRegex(ValueError, "either temporal_logps"):
            attribute_key_tokens(
                logits,
                original,
                visual,
                valid,
                temporal_logps=visual,
                temporal_delta_scores=temporal_scores,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
