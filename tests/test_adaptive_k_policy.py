#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only tests for the adaptive-k policy (no vLLM needed)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches"))

from adaptive_k_scheduler import (  # noqa: E402
    AdaptiveKConfig,
    AdaptiveKPolicy,
    DraftedRing,
    assign_lengths,
    filter_candidates,
    placeholder_len,
    should_observe,
)


def policy(**kw) -> AdaptiveKPolicy:
    return AdaptiveKPolicy(AdaptiveKConfig(**kw))


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = AdaptiveKConfig()
        self.assertEqual(c.k_set, (2, 4, 7))
        self.assertEqual((c.alpha, c.margin, c.min_steps), (0.25, 1.0, 4))
        self.assertEqual((c.saturate, c.mode, c.log_every), ("max", "batch-uniform", 200))
        self.assertTrue(c.enabled)

    def test_env_parsing(self):
        env = {
            "VLLM_ADAPTIVE_K_ENABLE": "1",
            "VLLM_ADAPTIVE_K_SET": "7,2,4,4",
            "VLLM_ADAPTIVE_K_ALPHA": "0.5",
            "VLLM_ADAPTIVE_K_MARGIN": "0.5",
            "VLLM_ADAPTIVE_K_MIN_STEPS": "8",
            "VLLM_ADAPTIVE_K_SATURATE": "n",
            "VLLM_ADAPTIVE_K_MODE": "per-request",
            "VLLM_ADAPTIVE_K_LOG_EVERY": "0",
        }
        c = AdaptiveKConfig.from_env(env)
        self.assertEqual(c.k_set, (2, 4, 7))
        self.assertEqual((c.alpha, c.margin, c.min_steps), (0.5, 0.5, 8))
        self.assertEqual((c.saturate, c.mode, c.log_every), ("n", "per-request", 0))
        self.assertFalse(AdaptiveKConfig.from_env({"VLLM_ADAPTIVE_K_ENABLE": "0"}).enabled)

    def test_validation(self):
        for bad in ((), (0, 2), (4, 2), (2, 2, 7)):
            with self.assertRaises(ValueError):
                AdaptiveKConfig(k_set=bad)
        with self.assertRaises(ValueError):
            AdaptiveKConfig(alpha=0.0)
        with self.assertRaises(ValueError):
            AdaptiveKConfig(margin=-0.1)
        with self.assertRaises(ValueError):
            AdaptiveKConfig(min_steps=-1)
        with self.assertRaises(ValueError):
            AdaptiveKConfig(saturate="ratchet")
        with self.assertRaises(ValueError):
            AdaptiveKConfig(mode="random")


class TestPolicy(unittest.TestCase):
    def test_new_request_starts_full(self):
        p = policy()
        self.assertEqual(p.decide("a"), 7)
        self.assertEqual(p.ema("a"), 7.0)

    def test_minimum_observations_stay_full(self):
        p = policy(alpha=1.0)
        for _ in range(3):
            p.observe("a", 0, 7)
            self.assertEqual(p.decide("a"), 7)
        p.observe("a", 0, 7)
        self.assertEqual(p.decide("a"), 2)

    def test_candidate_selection(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("two", 0, 7)
        p.observe("four", 3, 7)
        p.observe("seven", 6, 7)
        self.assertEqual(p.decide("two"), 2)
        self.assertEqual(p.decide("four"), 4)
        self.assertEqual(p.decide("seven"), 7)

    def test_ema_update(self):
        p = policy(alpha=0.5, min_steps=0)
        self.assertAlmostEqual(p.observe("a", 0, 7), 3.5)
        self.assertAlmostEqual(p.observe("a", 2, 7), 2.75)
        self.assertEqual(p.counters["observations"], 2)

    def test_full_short_prefix_can_climb(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("a", 0, 7)
        self.assertEqual(p.decide("a"), 2)
        p.observe("a", 2, 2)
        self.assertEqual(p.ema("a"), 7.0)
        self.assertEqual(p.decide("a"), 7)

    def test_saturate_n_ratchets(self):
        p = policy(alpha=1.0, min_steps=1, saturate="n")
        p.observe("a", 0, 7)
        self.assertEqual(p.decide("a"), 2)
        p.observe("a", 2, 2)
        self.assertEqual(p.ema("a"), 2.0)
        self.assertEqual(p.decide("a"), 2)

    def test_structured_request_pins_full(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("json", 0, 7)
        self.assertEqual(p.decide("json"), 2)
        self.assertEqual(p.decide("json", structured=True), 7)

    def test_batch_uniform_uses_minimum(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("high", 7, 7)
        p.observe("mid", 3, 7)
        p.observe("low", 0, 7)
        self.assertEqual(p.decide_batch(["high", "mid"]), 4)
        self.assertEqual(p.decide_batch(["high", "low"]), 2)
        self.assertEqual(p.decide_batch([]), 7)

    def test_structured_only_batch_stays_full(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("prose", 0, 7)
        p.observe("json", 0, 7)
        self.assertEqual(p.decide_batch(["json"], {"json"}), 7)
        self.assertEqual(p.decide_batch(["prose", "json"], {"json"}), 7)

    def test_uncalibrated_request_pins_batch_full(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("prose", 0, 7)
        self.assertEqual(p.decide_batch(["prose", "new"]), 7)

    def test_zero_drafts_is_no_op(self):
        p = policy()
        p.observe("a", 0, 0)
        self.assertEqual(p.ema("a"), 7.0)
        self.assertEqual(p.counters["observations"], 0)

    def test_evict_and_counters(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("a", 0, 7)
        p.decide("a")
        self.assertIn("k2=1", p.counters_line())
        p.evict("a")
        self.assertEqual(p.tracked(), 0)
        self.assertEqual(p.decide("a"), 7)


class TestObserveGating(unittest.TestCase):
    def test_real_drafts_and_output_are_observed(self):
        self.assertTrue(should_observe("a", {"a"}, 7, True, 1))

    def test_guards(self):
        self.assertFalse(should_observe("resumed", {"a"}, 7, True, 1))
        self.assertFalse(should_observe("a", {"a"}, 0, True, 1))
        self.assertFalse(should_observe("a", {"a"}, 7, False, 1))
        self.assertTrue(should_observe("a", {"a"}, 7, False, 0))
        self.assertFalse(should_observe("a", {"a"}, 7, True, 1, is_stale=True))
        self.assertFalse(should_observe("a", {"a"}, 7, True, 1, kv_load_failed=True))

    def test_ring_covers_async_lookahead(self):
        ring = DraftedRing(3)
        ring.push({"old"})
        ring.push({"mid"})
        ring.push({"new"})
        self.assertTrue(should_observe("old", ring.union(), 7, True, 1))
        ring.push({"newer"})
        self.assertFalse(should_observe("old", ring.union(), 7, True, 1))
        self.assertEqual(len(ring), 3)
        with self.assertRaises(ValueError):
            DraftedRing(0)


class TestPlaceholderSizing(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(placeholder_len(7, 7), 7)
        self.assertEqual(placeholder_len(7, 5), 5)
        self.assertEqual(placeholder_len(2, 7), 2)
        self.assertEqual(placeholder_len(-1, 7), 0)

    def test_filter_candidates(self):
        items = [
            ("running", False, False, True),
            ("prefill", False, True, True),
            ("empty", False, False, False),
            ("finished", True, False, True),
        ]
        self.assertEqual(filter_candidates(items), ["running"])

    def test_assign_lengths_batch_uniform(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("high", 7, 7)
        p.observe("mid", 3, 7)
        self.assertEqual(
            assign_lengths(p, ["high", "mid"], "batch-uniform", 7),
            {"high": 4, "mid": 4},
        )

    def test_assign_lengths_per_request(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("high", 7, 7)
        p.observe("low", 0, 7)
        self.assertEqual(
            assign_lengths(p, ["high", "low", "new"], "per-request", 7),
            {"high": 7, "low": 2, "new": 7},
        )

    def test_dynamic_table_choice_is_neutralised(self):
        p = policy(alpha=1.0, min_steps=1)
        p.observe("a", 7, 7)
        p.observe("b", 7, 7)
        self.assertEqual(
            assign_lengths(p, ["a", "b"], "batch-uniform", 7),
            {"a": 7, "b": 7},
        )


if __name__ == "__main__":
    unittest.main(verbosity=1)
