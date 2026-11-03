"""CPU-only regression tests for frozen search budget, failure and fairness invariants."""

import copy
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import search_core as sc


class Oracle:
    def __init__(self, costs=None, score=0.5):
        self.costs = costs or {None: 200, 1: 50, 2: 100}
        self.score = score
        self.calls = []

    def estimate(self, alpha, rung):
        return self.costs[rung]

    def evaluate(self, alpha, rung):
        self.calls.append(rung)
        return {"score": self.score, "new_samples": self.costs[rung]}


def build(method="random_box", budget=4400, oracle=None, config=None):
    cfg = dict(sc.DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    inits = [np.eye(10)[i % 10] for i in range(20)]
    oracle = oracle or Oracle()
    return sc.build_search(method, "full", 10, cfg, oracle,
                           np.random.default_rng(42), inits,
                           [0.1] * 20, [200] * 20, budget), oracle


class TestFrozenCoreExtra(unittest.TestCase):
    def test_init_budget_includes_shared_initialization(self):
        with self.assertRaises(sc.BudgetExhausted):
            build(budget=3999)

    def test_nonfinite_budget_rejected(self):
        with self.assertRaises(ValueError):
            build(budget=math.inf)

    def test_invalid_init_cost_and_score_rejected(self):
        cfg = dict(sc.DEFAULT_CONFIG)
        init = [np.zeros(10)] * 20
        for costs, scores in [([math.nan] + [200] * 19, [0.5] * 20),
                              ([200] * 20, [math.nan] + [0.5] * 19)]:
            with self.assertRaises(ValueError):
                sc.RandomBoxSearch(10, cfg, Oracle(), np.random.default_rng(1),
                                   init, scores, costs, 4400)

    def test_nonfinite_oracle_score_is_failure_not_champion(self):
        search, oracle = build(oracle=Oracle(score=math.nan))
        with self.assertRaises(ValueError):
            search.run(max_rounds=1)
        self.assertEqual(search.ledger.budget_used, 4200)
        self.assertEqual(search.ledger.events[-1]["kind"], "failed_observation")
        self.assertEqual(search.final_selection()[2], 0.1)
        self.assertEqual(search.ledger.budget_reserved, 0)

    def test_invalid_oracle_cost_does_not_poison_ledger(self):
        oracle = Oracle()
        search, _ = build(oracle=oracle)
        cid = search._register(np.ones(10), "random")
        oracle.evaluate = lambda alpha, rung: {"score": 0.5, "new_samples": math.nan}
        with self.assertRaises(RuntimeError):
            search._call(cid, None)
        self.assertEqual(search.ledger.budget_used, 4000)
        self.assertEqual(search.ledger.budget_reserved, 0)

    def test_random_partial_batch_stops_at_exact_budget(self):
        search, oracle = build(budget=4200)
        self.assertEqual(search.run(), "budget_exhausted")
        self.assertEqual(oracle.calls, [None])
        self.assertEqual(search.ledger.budget_used, 4200)
        self.assertEqual(len(search.ledger.candidates), 21)

    def test_random_rounds_continue_without_restarting_counter(self):
        search, oracle = build(budget=6400)
        self.assertEqual(search.run(max_rounds=1), "max_rounds")
        self.assertEqual(len(oracle.calls), 6)
        self.assertEqual(search.run(max_rounds=2), "max_rounds")
        self.assertEqual(len(oracle.calls), 12)

    def test_cma_sh_does_not_start_unfinishable_generation(self):
        search, oracle = build(method="cma_sh", budget=4500)
        self.assertEqual(search.run(max_rounds=1), "budget_exhausted")
        self.assertEqual(oracle.calls, [])
        self.assertEqual(search.generation, 0)
        self.assertEqual(len(search.ledger.candidates), 20)

    def test_cma_sh_complete_generation_has_all_promotions(self):
        search, oracle = build(method="cma_sh", budget=6000)
        self.assertEqual(search.run(max_rounds=1), "max_rounds")
        self.assertEqual(oracle.calls.count(1), 8)
        self.assertEqual(oracle.calls.count(2), 4)
        self.assertEqual(oracle.calls.count(None), 2)

    def test_spearman_uses_average_tie_ranks(self):
        pairs = [(1, 1), (1, 1), (2, 2), (3, 3)]
        self.assertAlmostEqual(sc.rankcorr(pairs), 1.0)
        self.assertAlmostEqual(sc.rankcorr(pairs[::-1]), 1.0)

    def test_checkpoint_trajectory_equivalence(self):
        for method in ("racer_frozen_v1", "cma_full", "cma_sh", "pso_box",
                       "random_box", "pso_sum1", "pso_sum1_scalegrid"):
            with self.subTest(method=method):
                budget = 15000
                whole, _ = build(method, budget=budget)
                whole.run(max_rounds=2)
                partial, _ = build(method, budget=budget)
                snapshots = []
                partial.on_checkpoint = lambda search: snapshots.append(search.snapshot_state())
                partial.run(max_rounds=1)
                self.assertTrue(snapshots)
                checkpoint = json.loads(json.dumps(snapshots[-1], allow_nan=False))
                restored, _ = build(method, budget=budget)
                restored.restore_state(checkpoint)
                if method != "pso_sum1_scalegrid":
                    restored.run(max_rounds=2)
                    self.assertEqual(restored.snapshot_state(), whole.snapshot_state())
                else:
                    # Grid is a terminal phase and must not execute twice.
                    before = len(restored.ledger.events)
                    restored.run(max_rounds=1)
                    self.assertEqual(len(restored.ledger.events), before)
                    self.assertEqual(restored.snapshot_state(), partial.snapshot_state())

    def test_checkpoint_rejects_digest_and_identity_mismatch(self):
        original, _ = build()
        original.run(max_rounds=1)
        snapshot = original.snapshot_state()
        altered, _ = build()
        broken = copy.deepcopy(snapshot)
        broken["ledger"]["budget_used"] += 1
        with self.assertRaisesRegex(ValueError, "digest"):
            altered.restore_state(broken)
        other_method, _ = build(method="pso_box")
        with self.assertRaisesRegex(ValueError, "identity"):
            other_method.restore_state(snapshot)
        other_budget, _ = build(budget=4800)
        with self.assertRaisesRegex(ValueError, "identity"):
            other_budget.restore_state(snapshot)

    def test_checkpoint_rejects_incomplete_round(self):
        search, _ = build(method="racer_frozen_v1", budget=15000)
        # A deliberately failed oracle does not yield a checkpointable round.
        search.oracle.score = math.nan
        with self.assertRaises(ValueError):
            search.run(max_rounds=1)
        with self.assertRaises(RuntimeError):
            search.snapshot_state()

    def test_scale_grid_sort_preserves_zero_score_order(self):
        search, oracle = build(method="pso_sum1_scalegrid", budget=7000)
        search.registry[0]["full"] = 0.0
        search.registry[1]["full"] = 0.2
        search.registry[2]["full"] = 0.1
        # First grid candidate must be derived from candidate 1, not a
        # lower-ranked candidate promoted by treating zero as false.
        search._scale_grid_phase()
        first = search.registry[20]["alpha"]
        expected = search._clip(search.registry[1]["alpha"] * 0.5)
        np.testing.assert_array_equal(first, expected)


if __name__ == "__main__":
    unittest.main()
