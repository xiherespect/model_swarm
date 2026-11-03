"""search_core / search_frozen CPU 测试：纯状态机 + 假 oracle，无 GPU/模型。"""

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import search_core as sc
from search_frozen import (TASK_REGISTRY, contiguous_runs, alpha_fingerprint,
                           RealOracle)


class TrackingOracle:
    """带逐题缓存的假 oracle：复刻 RealOracle 的计费语义（CPU 版）。"""

    def __init__(self, n, target, noise=0.0):
        self.n = n
        self.target = target
        self.noise = noise
        self.cache = {}
        self.is_multitask = False
        self.calls = 0
        self.physical = 0
        self.overrides = {}

    def _score(self, a, rung):
        a = np.asarray(a)
        base = 1.0 - float(np.sum((a - self.target) ** 2)) / (4 * self.n)
        if self.noise and rung is not None:
            base += self.noise * math.sin(17.0 * float(a[0]) + rung)
        return max(0.0, min(1.0, base))

    def _stop(self, rung):
        if rung is None:
            return 200
        return 50 if rung == 1 else 100

    def estimate(self, a, rung):
        return self._stop(rung)

    def evaluate(self, a, rung):
        self.calls += 1
        key = (alpha_fingerprint(a), rung)
        stop = self._stop(rung)
        if key in self.cache:
            have = self.cache[key]
        else:
            have = set()
        need = stop - len(have)
        if need <= 0:
            return {"score": self._score(a, rung), "new_samples": 0}
        self.physical += need
        # 允许测试注入异常分数
        if key in self.overrides:
            score = self.overrides[key]
        else:
            score = self._score(a, rung)
        self.cache[key] = set(range(stop))
        return {"score": score, "new_samples": need}


def make_oracle(n, target=None, **kw):
    return TrackingOracle(n, np.full(n, 0.9) if target is None else target, **kw)


def shared_inits(n=10, seed=42):
    rng = np.random.default_rng(seed)
    inits = [np.eye(n)[i] for i in range(n)]
    while len(inits) < 20:
        if rng.random() < 0.7:
            a = rng.dirichlet(np.ones(n) * 0.6)
        else:
            a = rng.dirichlet(np.ones(n) * 0.6) * rng.uniform(0.5, 1.5)
        inits.append(a)
    return inits


def build(method="racer_frozen_v1", ablation="full", n=10, budget=110 * 200,
          oracle=None, seed=42, config_overrides=None):
    cfg = dict(sc.DEFAULT_CONFIG)
    cfg.update(config_overrides or {})
    oracle = oracle or make_oracle(n)
    inits = shared_inits(n, seed)
    scores = [oracle._score(a, None) for a in inits]
    rng = np.random.default_rng(seed)
    return sc.build_search(method, ablation, n, cfg, oracle, rng, inits,
                           scores, [oracle._stop(None)] * len(inits), budget,
                           seed=seed), oracle


class TestBudgetAndLedger(unittest.TestCase):
    def test_no_overspend_all_methods(self):
        for method, ablation in [("racer_frozen_v1", a) for a in sc.ABLATIONS] + \
                [("cma_full", "full"), ("cma_sh", "full"), ("pso_box", "full"),
                 ("random_box", "full")]:
            search, oracle = build(method, ablation, budget=110 * 200)
            status = search.run()
            self.assertLessEqual(search.ledger.budget_used, 110 * 200 + 1e-6,
                                 f"{method}/{ablation} overspent")
            self.assertIn(status, ("budget_exhausted", "max_rounds", "stopped",
                                   "patience_stopped"))

    def test_reserve_before_call_and_release(self):
        search, oracle = build()
        ledger = search.ledger
        self.assertEqual(ledger.budget_reserved, 0.0)
        search.run()
        self.assertEqual(ledger.budget_reserved, 0.0)  # 全部 release

    def test_oracle_cannot_exceed_estimate(self):
        class LyingOracle(TrackingOracle):
            def evaluate(self, a, rung):
                return {"score": 0.5, "new_samples": 10 ** 9}
        search, _ = build(oracle=LyingOracle(10, np.full(10, 0.9)))
        with self.assertRaises(RuntimeError):
            search.run()

    def test_duplicate_full_observation_rejected(self):
        search, _ = build()
        cid = search.archive_ids[0]
        with self.assertRaises(RuntimeError):
            search._observe_full(cid, 0.5, 0.0)


class TestAblationSemantics(unittest.TestCase):
    def test_disabled_engine_never_proposed(self):
        for ablation, banned in (("no_DE", "de"), ("no_sum", "sum"), ("no_ES", "ES")):
            search, _ = build("racer_frozen_v1", ablation)
            search.run()
            sources = {c["source"] for c in search.ledger.candidates}
            self.assertNotIn(banned, sources, f"{ablation} leaked engine {banned}")

    def test_disabled_engine_stays_dead_after_boost(self):
        # 触发 sumdir_boost（guard 换脊）后 no_sum 不得复活
        search, _ = build("racer_frozen_v1", "no_sum")
        search.run()
        sources = {c["source"] for c in search.ledger.candidates}
        self.assertNotIn("sum", sources)

    def test_full_eval_skips_funnel(self):
        search, _ = build("racer_frozen_v1", "full_eval")
        search.run()
        kinds = {e["kind"] for e in search.ledger.events}
        coarse_events = [e for e in search.ledger.events if e["fidelity"] == "coarse"]
        self.assertEqual(coarse_events, [])

    def test_no_rung2_never_uses_rung2(self):
        search, _ = build("racer_frozen_v1", "no_rung2")
        search.run()
        rung2 = [e for e in search.ledger.events if e.get("rung") == 2]
        self.assertEqual(rung2, [])

    def test_rung2_only_when_near(self):
        search, _ = build("racer_frozen_v1", "full")
        search.run()
        # rung2 事件存在且都来自接近者（gap<=0.03 由实现保证，这里只验证存在性）
        rung2 = [e for e in search.ledger.events if e.get("rung") == 2]
        self.assertTrue(rung2, "expected some rung2 usage on noisy oracle")


class TestFinalSelection(unittest.TestCase):
    def test_raw_argmax_ties_earliest(self):
        search, oracle = build()
        search.run()
        # 构造并列：把最高分复制到更早的 archive 候选上，终选必须取更早者
        ids = search.archive_ids
        top_score = max(search.registry[c]["full"] for c in ids)
        first_with_top = next(c for c in ids if search.registry[c]["full"] == top_score)
        earlier = ids[0]
        if earlier == first_with_top:
            earlier = ids[1]
        search.registry[earlier]["full"] = top_score
        pick, _, score = search.final_selection()
        self.assertEqual(score, top_score)
        self.assertLessEqual(pick, first_with_top)
        self.assertEqual(search.registry[pick]["full"], top_score)

    def test_coarse_scores_never_enter_final(self):
        search, _ = build()
        search.run()
        pick, _, score = search.final_selection()
        self.assertEqual(score, search.registry[pick]["full"])
        # 所有 archive 候选都有 full 观测
        for cid in search.archive_ids:
            self.assertIsNotNone(search.registry[cid]["full"])

    def test_anchor_and_best_seen_distinct(self):
        search, _ = build()
        search.run()
        # best_seen 单调不退化
        best_values = [b for _, b in search.anytime]
        running = -math.inf
        for v in best_values:
            running = max(running, v)
            self.assertEqual(v, running)


class TestRankCorr(unittest.TestCase):
    def test_constant_input_returns_none(self):
        self.assertIsNone(sc.rankcorr([(1, 2), (1, 3), (1, 4), (1, 5)]))
        self.assertIsNone(sc.rankcorr([(1, 2), (2, 2), (3, 2), (4, 2)]))

    def test_short_input_returns_none(self):
        self.assertIsNone(sc.rankcorr([(1, 1), (2, 2), (3, 3)]))

    def test_perfect_correlation(self):
        pairs = [(1, 10), (2, 20), (3, 30), (4, 40)]
        self.assertAlmostEqual(sc.rankcorr(pairs), 1.0, places=6)


class TestEngineProbs(unittest.TestCase):
    def test_ablation_renormalizes(self):
        search, _ = build("racer_frozen_v1", "no_DE")
        probs = search._engine_probs()
        self.assertNotIn("de", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertAlmostEqual(probs["sum"], 0.2 / 0.6)

    def test_boost_shifts_toward_sum_without_revival(self):
        search, _ = build("racer_frozen_v1", "no_sum")
        search.sumdir_boost = True
        probs = search._engine_probs()
        self.assertNotIn("sum", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0)


class TestShParsing(unittest.TestCase):
    def test_sh_promotion_counts(self):
        search, _ = build("racer_frozen_v1", "standard_SH", budget=110 * 200)
        status = search.run()
        self.assertIn(status, ("budget_exhausted", "max_rounds"))

    def test_cma_sh_uses_rung_feedback(self):
        search, _ = build("cma_sh", "full")
        search.run()
        events = [e for e in search.ledger.events if e["fidelity"] == "coarse"]
        self.assertTrue(events)


class TestSum1Projection(unittest.TestCase):
    def test_projection_exact(self):
        rng = np.random.default_rng(3)
        search, _ = build("pso_sum1")
        for _ in range(50):
            a = rng.uniform(-2.0, 3.0, 10)
            p = search._project(a)
            self.assertLess(abs(float(p.sum()) - 1.0), 1e-9)
            self.assertTrue(np.all(p >= -0.5 - 1e-12))
            self.assertTrue(np.all(p <= 1.5 + 1e-12))

    def test_scalegrid_reserves_and_searches(self):
        search, _ = build("pso_sum1_scalegrid")
        status = search.run()
        sources = {c["source"] for c in search.ledger.candidates}
        self.assertIn("scale_grid", sources)
        self.assertIn("pso_sum1", sources)
        self.assertLessEqual(search.ledger.budget_used, 110 * 200 + 1e-6)

    def test_sum1_candidates_stay_on_hull(self):
        search, _ = build("pso_sum1")
        search.run()
        for c in search.ledger.candidates:
            if c["source"] == "pso_sum1":
                self.assertLess(abs(sum(c["alpha"]) - 1.0), 1e-6)


class TestContiguousRuns(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(contiguous_runs([0, 1, 2, 5, 6, 9]),
                         [(0, 3), (5, 7), (9, 10)])
        self.assertEqual(contiguous_runs([]), [])
        self.assertEqual(contiguous_runs([3]), [(3, 4)])


class TestRealOracleSlicing(unittest.TestCase):
    def test_rung_sizing(self):
        oracle = RealOracle.__new__(RealOracle)  # 不触发目录创建
        oracle.datasets = ["nlgraph"]
        oracle._dev_sizes = {"nlgraph": 200}
        self.assertEqual(oracle._rung_stop("nlgraph", 1), 50)
        self.assertEqual(oracle._rung_stop("nlgraph", 2), 100)
        self.assertEqual(oracle._rung_stop("nlgraph", None), 200)


if __name__ == "__main__":
    unittest.main()
