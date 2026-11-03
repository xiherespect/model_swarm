"""search_core.py — racer_frozen_v1 纯状态搜索核心（NumPy + 标准库；CMA 驱动惰性 import pycma）。

从 search_racer.py / search.py / search_baseline.py 移植明确公式并冻结：
不 import 旧模块、不 exec 旧脚本、无任何 IO/GPU/模型操作。算法状态完全由
(冻结配置, RNG 状态, 事件账本) 决定；物化与评测通过注入的 oracle 回调进行。

Oracle 接口（由 search_frozen.py 绑定真实物化+评测）:
    oracle.evaluate(alpha, rung) -> {"score": float, "new_samples": int}
        rung: None=完整dev / 1=粗评(每组件前 min(50,n) 条) / 2=二级(每组件前 min(100,n))
        逐题增量缓存由 oracle 负责；new_samples 只计真实新增推理样本。
    oracle.estimate(alpha, rung) -> int
        该调用最坏新增样本数（缓存命中只会更少）。核心在发调用前用它查预算。

冻结语义（相对旧实现修复的缺陷不再继承）:
  - no_de/no_sum/no_ES 在普通路径真实生效；禁用引擎经重启/boost 不复活。
  - anchor（搜索锚，guard 可换）与 best_seen（历史最优，永不退化）分开记录。
  - 终选 = 完整 dev 首次观测分数 raw argmax，并列取最早 candidate ID；无 LCB、
    无冠军复评、无 holdout 分支。
  - 预算以样本数计（B_t = 110×D_t），发调用前 reserve，实际用量 commit。
  - focus/highway/stretch/adaptive_sum/代理LCB/复评 均不移植。
"""

import copy
import hashlib
import json
import math

import numpy as np

RACER_VERSION = "racer_frozen_v1"
CORE_SPEC_VERSION = 1
METHODS = ("racer_frozen_v1", "cma_full", "pso_box", "random_box", "cma_sh")
ENGINE_ABLATIONS = ("full", "no_DE", "no_sum", "no_ES")
ABLATIONS = ENGINE_ABLATIONS + ("no_guard", "no_plateau", "no_rung2",
                                "full_eval", "standard_SH")

DEFAULT_CONFIG = {
    "lb": -0.5, "ub": 1.5,
    "init_size": 20,
    "round_evals": 4,              # k：每轮真实评估名额
    "proposal_multiplier": 3,      # Q = 3k
    "coarse_size": 50,             # rung1 每组件样本
    "rung2_size": 100,             # rung2 每组件样本
    "engine_probs": {"ES": 0.4, "de": 0.4, "sum": 0.2},
    "sum_lo": 1.2, "sum_hi": 1.9,
    "elite_percent": 0.25,
    "guard_patience": 3,
    "guard_signature": {"sum_lt": 1.15, "k_le": 3, "top2_gt": 0.8, "support_eps": 0.05},
    "ridge_margin": 0.02, "ridge_relax": 0.05,
    "plateau_patience": 2,
    "plateau_eps_single": 0.14, "plateau_eps_multitask": 0.10,
    "plateau_rho": 0.2, "rankcorr_min_pairs": 4,
    "rung2_gap": 0.03,
    "stagnation_restart": 3,
    "sigma_up": 1.1, "sigma_down": 0.85, "sigma_min": 0.15, "sigma_max": 3.0,
    "sigma_guard": 1.2, "sigma_restart": 1.5,
    "de_F": 0.5, "de_CR": 0.9, "shade_p": 0.11, "shade_memory": 5,
    "dedup_eps": 0.05, "sum_perturb": 0.05,
    "es_sd_lo": 0.02, "es_sd_hi": 0.5, "es_path_coeff": 0.15,
    "evo_decay": 0.7, "evo_gain": 0.3,
    "elite_weight_hi": 1.0, "elite_weight_lo": 0.3,
    "sh_eta": 0.5, "sh_min_promote": 1,
    "cma": {"popsize": 8, "sigma0": 0.3},
    "cma_sh": {"popsize": 8, "sigma0": 0.3, "rung2_promote": 4, "full_promote": 2},
    "pso": {"inertia": 0.4, "cognitive": 0.3, "social": 0.3, "repel": 0.3,
            "step_length": 1.0, "step_factor": 0.95, "min_step": 0.1,
            "patience": 10, "restart_patience": 0.5},
    "random_batch": 6, "random_dedup_tries": 10,
}


class BudgetExhausted(Exception):
    """剩余预算不足以发出下一个调用；停止并报告，不得超支。"""


def stable_argmax(values):
    values = list(values)
    if not values:
        raise ValueError("argmax requires at least one value")
    return max(range(len(values)), key=values.__getitem__)


def top2_mass(a):
    a = np.abs(np.asarray(a, dtype=np.float64))
    return float(np.sort(a)[::-1][:2].sum() / (a.sum() + 1e-12))


def top2_pair(a):
    return tuple(sorted(np.argsort(np.abs(np.asarray(a, dtype=np.float64)))[::-1][:2].tolist()))


def rankcorr(pairs):
    """Spearman 秩相关；<4 对或任一侧为常数时返回 None（不伪造值）。"""
    if len(pairs) < 4:
        return None
    x = np.array([p[0] for p in pairs], dtype=np.float64)
    y = np.array([p[1] for p in pairs], dtype=np.float64)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    def average_ranks(values):
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks

    rx, ry = average_ranks(x), average_ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho if math.isfinite(rho) else None


def validate_config(config, budget_samples):
    if (not isinstance(budget_samples, (int, float))
            or not math.isfinite(budget_samples) or budget_samples <= 0):
        raise ValueError("budget_samples must be finite and positive")
    missing = [key for key in DEFAULT_CONFIG if key not in config]
    if missing:
        raise ValueError("config missing frozen keys: " + ", ".join(missing))
    probs = config["engine_probs"]
    if set(probs) != {"ES", "de", "sum"} or abs(sum(probs.values()) - 1.0) > 1e-9:
        raise ValueError("engine_probs must cover ES/de/sum and sum to 1")
    for engine, value in probs.items():
        if not 0 < value <= 1:
            raise ValueError("engine probabilities must be in (0, 1]")
    if not 2 <= config["round_evals"] <= 12:
        raise ValueError("round_evals out of frozen range")
    if not 0 < config["coarse_size"] < config["rung2_size"]:
        raise ValueError("rung sizes must satisfy 0 < rung1 < rung2")
    if not 0 < config["sh_eta"] < 1:
        raise ValueError("sh_eta must be in (0, 1)")
    return True


def _engine_set(ablation):
    if ablation == "no_DE":
        return ("ES", "sum")
    if ablation == "no_sum":
        return ("ES", "de")
    if ablation == "no_ES":
        return ("de", "sum")
    return ("ES", "de", "sum")


class Ledger:
    """append-only 事件账本；reserve 先于调用，commit 实际用量。"""

    def __init__(self):
        self.candidates = []    # {"id","alpha","source"}
        self.events = []        # {"kind","candidate_id","fidelity","rung","samples","score"}
        self.budget_used = 0.0
        self.budget_reserved = 0.0

    def reserve(self, samples):
        if samples < 0:
            raise ValueError("reserve must be nonnegative")
        self.budget_reserved += samples

    def commit(self, samples, event):
        if samples < 0:
            raise ValueError("commit must be nonnegative")
        self.budget_used += samples
        self.budget_reserved = max(0.0, self.budget_reserved - samples)
        self.events.append(event)

    def release(self, samples):
        self.budget_reserved = max(0.0, self.budget_reserved - samples)


class _BaseSearch:
    """共同骨架：初始化计费、档案（仅完整观测入档案）、anytime 轨迹、共同终选。"""

    def __init__(self, n, config, oracle, rng, init_alphas, init_scores,
                 init_costs, budget_samples, method, ablation="full", seed=42):
        validate_config(config, budget_samples)
        if ablation not in ABLATIONS:
            raise ValueError("unknown ablation: " + str(ablation))
        self.seed = seed
        self.n = n
        self.config = dict(config)
        self.oracle = oracle
        self.rng = rng
        self.method = method
        self.ablation = ablation
        self.budget_samples = budget_samples
        self.ledger = Ledger()
        self.registry = {}       # cid -> {"alpha","source","full","coarse"}
        self.archive_ids = []    # 完整观测候选 ID（入档案序）
        self.best_seen_id = None
        self.best_seen_score = -math.inf
        self.anytime = []        # (budget_used, best_seen_score) 每事件后
        if len(init_alphas) != len(init_scores) or len(init_alphas) != len(init_costs):
            raise ValueError("init triples must align")
        if len(init_alphas) != self.config["init_size"]:
            raise ValueError("init size must equal frozen init_size")
        if any(not math.isfinite(float(cost)) or cost < 0 for cost in init_costs):
            raise ValueError("init costs must be finite and nonnegative")
        if sum(init_costs) > budget_samples + 1e-9:
            raise BudgetExhausted("shared initialization exceeds search budget")
        self.on_checkpoint = None  # callback(search), at completed round/generation only
        self._at_boundary = False
        for alpha, score, cost in zip(init_alphas, init_scores, init_costs):
            if not math.isfinite(float(score)):
                raise ValueError("init scores must be finite")
            cid = self._register(np.asarray(alpha, dtype=np.float64), "init")
            self._observe_full(cid, float(score), float(cost))
        self._at_boundary = True

    # ---------- 档案 ----------
    def _register(self, alpha, source, meta=None):
        cid = len(self.ledger.candidates)
        alpha = np.asarray(alpha, dtype=np.float64).copy()
        if alpha.shape != (self.n,):
            raise ValueError("alpha dimension mismatch")
        self.registry[cid] = {"alpha": alpha, "source": source, "full": None,
                              "coarse": {}}
        if meta:
            self.registry[cid].update(meta)
        self.ledger.candidates.append({"id": cid, "alpha": alpha.tolist(),
                                       "source": source})
        return cid

    def _observe_full(self, cid, score, samples):
        if self.registry[cid]["full"] is not None:
            raise RuntimeError("duplicate full observation for candidate")
        self.registry[cid]["full"] = score
        self.archive_ids.append(cid)
        if score > self.best_seen_score:
            self.best_seen_id, self.best_seen_score = cid, score
        self.ledger.commit(samples, {"kind": "observe", "candidate_id": cid,
                                     "fidelity": "full", "rung": None,
                                     "samples": samples, "score": score})
        self.anytime.append((self.ledger.budget_used, self.best_seen_score))

    def _observe_coarse(self, cid, rung, score, samples):
        if rung in self.registry[cid]["coarse"]:
            raise RuntimeError("duplicate coarse observation for candidate/rung")
        self.registry[cid]["coarse"][rung] = score
        self.ledger.commit(samples, {"kind": "observe", "candidate_id": cid,
                                     "fidelity": "coarse", "rung": rung,
                                     "samples": samples, "score": score})
        self.anytime.append((self.ledger.budget_used, self.best_seen_score))

    def _budget_left(self):
        return self.budget_samples - self.ledger.budget_used

    def _call(self, cid, rung):
        """预算先检后调：estimate 超剩余即停；commit 实际用量。"""
        alpha = self.registry[cid]["alpha"]
        estimate = self.oracle.estimate(alpha, rung)
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError("oracle estimate must be finite and nonnegative")
        if estimate > self._budget_left() + 1e-9:
            raise BudgetExhausted(
                f"need<={self._budget_left():.1f} samples, next call needs {estimate}")
        self.ledger.reserve(estimate)
        try:
            result = self.oracle.evaluate(alpha, rung)
        finally:
            self.ledger.release(estimate)
        samples = float(result["new_samples"])
        score = float(result["score"])
        if not math.isfinite(samples) or samples < 0 or samples > estimate + 1e-9:
            raise RuntimeError("oracle returned invalid sample cost or exceeded its estimate")
        if not math.isfinite(score):
            # Samples were physically consumed even if the score is invalid.
            self.ledger.commit(samples, {"kind": "failed_observation", "candidate_id": cid,
                                         "fidelity": "full" if rung is None else "coarse",
                                         "rung": rung, "samples": samples, "score": None})
            self.anytime.append((self.ledger.budget_used, self.best_seen_score))
            raise ValueError("oracle score must be finite")
        if rung is None:
            self._observe_full(cid, score, samples)
        else:
            self._observe_coarse(cid, rung, score, samples)
        return score, samples

    def _clip(self, a):
        return np.clip(np.asarray(a, dtype=np.float64),
                       self.config["lb"], self.config["ub"])

    # ---------- 共同终选 ----------
    def final_selection(self):
        """完整 dev 首次观测 raw argmax；并列最早 candidate ID。"""
        if not self.archive_ids:
            raise RuntimeError("no full observations")
        scores = [self.registry[cid]["full"] for cid in self.archive_ids]
        pick = self.archive_ids[stable_argmax(scores)]
        return pick, self.registry[pick]["alpha"].copy(), self.registry[pick]["full"]

    def anytime_at(self, marks):
        """各预算标记处的 best_seen（同一轨迹的检查点，非独立实验）。"""
        out = []
        for mark in marks:
            value = None
            for budget, best in self.anytime:
                if budget <= mark + 1e-9:
                    value = best
                else:
                    break
            out.append(value)
        return out

    def _snapshot_base(self):
        return {"method": self.method, "ablation": self.ablation,
                "budget_samples": self.budget_samples,
                "budget_used": self.ledger.budget_used,
                "best_seen_id": self.best_seen_id,
                "best_seen_score": self.best_seen_score,
                "candidates": self.ledger.candidates,
                "events": self.ledger.events}

    def _checkpoint(self):
        self._at_boundary = True
        if self.on_checkpoint is not None:
            self.on_checkpoint(self)

    def _state_fields(self):
        return ()

    def snapshot_state(self):
        """JSON-safe completed-round checkpoint; never a partial generation."""
        if not self._at_boundary or self.ledger.budget_reserved:
            raise RuntimeError("checkpoint requires completed round and no reserved samples")
        registry = []
        for cid in range(len(self.ledger.candidates)):
            entry = self.registry[cid]
            registry.append({"alpha": entry["alpha"].tolist(),
                             "source": entry["source"], "full": entry["full"],
                             "coarse": {str(k): v for k, v in entry["coarse"].items()},
                             **{key: entry[key] for key in ("de_F", "de_CR") if key in entry}})
        state = {"version": 1, "method": self.method, "ablation": self.ablation,
                 "n": self.n, "seed": self.seed, "budget_samples": self.budget_samples,
                 "config": self.config, "rng": self.rng.bit_generator.state,
                 "ledger": self._snapshot_base(), "registry": registry,
                 "archive_ids": self.archive_ids, "best_seen_id": self.best_seen_id,
                 "best_seen_score": self.best_seen_score, "anytime": self.anytime,
                 "optimizer": {key: getattr(self, key) for key in self._state_fields()}}
        def jsonable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {str(k): jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [jsonable(v) for v in obj]
            return obj
        state = jsonable(state)
        # A checksum catches torn/accidentally mismatched state, not adversarial changes.
        state["digest"] = hashlib.sha256(json.dumps(state, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()
        return state

    def restore_state(self, state):
        """Restore matching core state after runner restores the oracle cache.

        Caller must construct with saved init triples (no init evaluation) and
        restore the paired oracle cache before invoking run(). Checkpoints from
        an incomplete round are not supported.
        """
        state = copy.deepcopy(state)
        digest = state.pop("digest", None)
        if not isinstance(digest, str) or hashlib.sha256(json.dumps(
                state, sort_keys=True, allow_nan=False).encode()).hexdigest() != digest:
            raise ValueError("checkpoint digest mismatch")
        for key, expected in (("version", 1), ("method", self.method),
                              ("ablation", self.ablation), ("n", self.n),
                              ("seed", self.seed), ("budget_samples", self.budget_samples),
                              ("config", self.config)):
            if state[key] != expected:
                raise ValueError("checkpoint identity mismatch: " + key)
        ledger = state["ledger"]
        registry = state["registry"]
        candidates = ledger["candidates"]
        if len(candidates) != len(registry) or len(candidates) < self.config["init_size"]:
            raise ValueError("checkpoint candidate count mismatch")
        if candidates[:self.config["init_size"]] != self.ledger.candidates:
            raise ValueError("checkpoint initialization mismatch")
        if any(c["id"] != i or c["alpha"] != r["alpha"] or
               c["source"] != r["source"] or len(r["alpha"]) != self.n
               for i, (c, r) in enumerate(zip(candidates, registry))):
            raise ValueError("checkpoint registry mismatch")
        if (ledger["budget_used"] > self.budget_samples + 1e-9 or
                abs(sum(e["samples"] for e in ledger["events"]) -
                    ledger["budget_used"]) > 1e-6):
            raise ValueError("checkpoint budget mismatch")
        archive_ids = state["archive_ids"]
        if (archive_ids != [i for i, r in enumerate(registry) if r["full"] is not None]
                or not archive_ids):
            raise ValueError("checkpoint archive mismatch")
        best = min(archive_ids, key=lambda i: (-registry[i]["full"], i))
        if state["best_seen_id"] != best or state["best_seen_score"] != registry[best]["full"]:
            raise ValueError("checkpoint champion mismatch")
        if set(state["optimizer"]) != set(self._state_fields()):
            raise ValueError("checkpoint optimizer fields mismatch")
        self.ledger.candidates = candidates
        self.ledger.events = ledger["events"]
        self.ledger.budget_used = ledger["budget_used"]
        self.ledger.budget_reserved = 0.0
        self.registry = {i: {**r, "alpha": np.asarray(r["alpha"], dtype=np.float64),
                             "coarse": {int(k): v for k, v in r["coarse"].items()}}
                         for i, r in enumerate(registry)}
        self.archive_ids = archive_ids
        self.best_seen_id = best
        self.best_seen_score = state["best_seen_score"]
        self.anytime = [tuple(x) for x in state["anytime"]]
        for key in self._state_fields():
            old = getattr(self, key)
            value = state["optimizer"][key]
            if isinstance(old, np.ndarray):
                value = np.asarray(value, dtype=old.dtype)
            elif isinstance(old, list) and old and isinstance(old[0], np.ndarray):
                value = [np.asarray(x, dtype=old[0].dtype) for x in value]
            elif key == "visited_pairs" or key == "coarse_full_pairs":
                value = [tuple(x) for x in value]
            setattr(self, key, value)
        self.rng.bit_generator.state = state["rng"]
        self._restore_optimizer()
        self._at_boundary = True

    def _restore_optimizer(self):
        pass


class RacerSearch(_BaseSearch):
    """三引擎 + 竞速漏斗 + guard/plateau/rung2 的完整 RACER。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="racer_frozen_v1", **kwargs)
        self.anchor_id = self.best_seen_id
        self.sigma = 1.0
        self.evo_path = np.zeros(self.n)
        memory = self.config["shade_memory"]
        self.M_F = [self.config["de_F"]] * memory
        self.M_CR = [self.config["de_CR"]] * memory
        self.mem_ptr = 0
        self.collapse_rounds = 0
        self.visited_pairs = []
        self.plateau_rounds = 0
        self.plateau_mode = False
        self.coarse_full_pairs = []
        self.stagnation = 0
        self.sumdir_boost = False
        self.round_index = 0

    def _state_fields(self):
        return ("anchor_id", "sigma", "evo_path", "M_F", "M_CR", "mem_ptr",
                "collapse_rounds", "visited_pairs", "plateau_rounds",
                "plateau_mode", "coarse_full_pairs", "stagnation",
                "sumdir_boost", "round_index")

    # ---------- 引擎概率（禁用真实生效；重启/boost 不复活）----------
    def _engine_probs(self):
        base = dict(self.config["engine_probs"])
        engines = _engine_set(self.ablation)
        probs = {e: base[e] for e in engines}
        if self.sumdir_boost and "sum" in probs:
            # 冻结规则：非 sum 引擎 ×0.75，让出质量给 sum；无 sum 则 no-op。
            freed = sum(probs[e] for e in probs if e != "sum") * 0.25
            probs = {e: (v * 0.75 if e != "sum" else v + freed)
                     for e, v in probs.items()}
        total = sum(probs.values())
        return {e: v / total for e, v in probs.items()}

    def _elites(self):
        k = max(3, int(self.config["elite_percent"] * len(self.archive_ids)))
        k = min(k, len(self.archive_ids))
        alphas = [self.registry[cid]["alpha"] for cid in self.archive_ids]
        scores = [self.registry[cid]["full"] for cid in self.archive_ids]
        # 降序、并列取最早 candidate ID
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [alphas[i] for i in order], [scores[i] for i in order]

    def _anchor_alpha(self):
        return self.registry[self.anchor_id]["alpha"]

    def _anchor_score(self):
        return self.registry[self.anchor_id]["full"]

    # ---------- 主循环 ----------
    def run(self, max_rounds=None):
        while max_rounds is None or self.round_index < max_rounds:
            self._at_boundary = False
            try:
                progressed = self._round()
            except BudgetExhausted:
                # The preflight rolls back proposals/guard before any call.
                # Unexpected mid-round stops remain non-checkpointable.
                if all(r["full"] is not None for r in self.registry.values()):
                    self._checkpoint()
                return "budget_exhausted"
            if not progressed:
                return "stopped"
            self.round_index += 1
            self._checkpoint()
        self._checkpoint()
        return "max_rounds"

    def _guard(self):
        cfg = self.config
        if self.ablation == "no_guard":
            self.collapse_rounds = 0
            return
        sig = cfg["guard_signature"]
        a = self._anchor_alpha()
        g_sum = float(np.sum(a))
        g_k = int((np.abs(a) > sig["support_eps"]).sum())
        if g_sum < sig["sum_lt"] or g_k <= sig["k_le"] or top2_mass(a) > sig["top2_gt"]:
            self.collapse_rounds += 1
        else:
            self.collapse_rounds = 0
        if self.collapse_rounds < cfg["guard_patience"]:
            return
        gpair = top2_pair(a)
        alphas = [self.registry[cid]["alpha"] for cid in self.archive_ids]
        scores = [self.registry[cid]["full"] for cid in self.archive_ids]
        pairs = [top2_pair(x) for x in alphas]
        margin = self._anchor_score() - cfg["ridge_margin"]
        cand = [i for i, p in enumerate(pairs)
                if p != gpair and p not in self.visited_pairs and scores[i] >= margin]
        if not cand:  # 放宽：允许分稍低但脊必须新
            relax = self._anchor_score() - cfg["ridge_relax"]
            cand = [i for i, p in enumerate(pairs)
                    if p != gpair and p not in self.visited_pairs and scores[i] >= relax]
        if not cand:  # 黑名单清空重探
            cand = [i for i, p in enumerate(pairs)
                    if p != gpair and scores[i] >= margin]
        if cand:
            j = max(cand, key=lambda i: scores[i])
            self.visited_pairs.append(gpair)
            self.visited_pairs = self.visited_pairs[-3:]
            self.anchor_id = self.archive_ids[j]  # 换 anchor；best_seen 不动
            self.ledger.events.append({"kind": "ridge_switch", "candidate_id": None,
                                       "fidelity": None, "rung": None, "samples": 0.0,
                                       "score": scores[j], "new_anchor_id": self.anchor_id})
        self.sigma = max(self.sigma, cfg["sigma_guard"])
        self.sumdir_boost = True
        self.collapse_rounds = 0

    def _propose(self, count):
        """按当前引擎概率生成 count 个提案（已 clip），返回 (alpha, engine) 列表。"""
        cfg = self.config
        E, _ = self._elites()
        probs = self._engine_probs()
        names = list(probs)
        pvals = [probs[name] for name in names]
        stack = np.stack(E)
        weights = np.linspace(cfg["elite_weight_hi"], cfg["elite_weight_lo"], len(E))
        weights = weights / weights.sum()
        m_es = np.average(stack, axis=0, weights=weights)
        sd_es = np.clip(stack.std(axis=0), cfg["es_sd_lo"], cfg["es_sd_hi"])
        alphas = [self.registry[cid]["alpha"] for cid in self.archive_ids]
        scores = [self.registry[cid]["full"] for cid in self.archive_ids]
        out = []
        for _ in range(count):
            engine = names[int(self.rng.choice(len(names), p=pvals))]
            meta = {}
            if engine == "ES":
                a = (m_es + self.sigma * sd_es * self.rng.standard_normal(self.n)
                     + cfg["es_path_coeff"] * self.evo_path)
            elif engine == "de":
                F = float(np.clip(self.rng.standard_cauchy() * 0.1
                                  + self.M_F[int(self.rng.integers(len(self.M_F)))],
                                  1e-3, 1.0))
                CR = float(np.clip(self.rng.normal(
                    self.M_CR[int(self.rng.integers(len(self.M_CR)))], 0.1), 0.0, 1.0))
                pbn = max(2, int(cfg["shade_p"] * len(alphas)))
                top = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:pbn]
                pbest = alphas[int(top[int(self.rng.integers(pbn))])]
                r1 = alphas[int(self.rng.integers(len(alphas)))]
                r2 = alphas[int(self.rng.integers(len(alphas)))]
                lam = alphas[int(self.rng.integers(len(alphas)))]
                v = lam + F * (pbest - lam) + F * (r1 - r2)
                cross = self.rng.random(self.n) < CR
                cross[int(self.rng.integers(self.n))] = True
                a = np.where(cross, v, lam)
                meta = {"de_F": F, "de_CR": CR}
            else:  # sum 定向
                anchor = E[int(self.rng.integers(len(E)))]
                target = float(self.rng.uniform(cfg["sum_lo"], cfg["sum_hi"]))
                a = anchor * (target / max(float(anchor.sum()), 0.3)) \
                    + self.rng.uniform(-cfg["sum_perturb"], cfg["sum_perturb"], self.n)
            out.append((self._clip(a), engine, meta))
        return out

    def _near_archive(self, a):
        eps = self.config["dedup_eps"]
        return any(np.linalg.norm(a - self.registry[cid]["alpha"]) < eps
                   for cid in self.archive_ids)

    def _round(self):
        cfg = self.config
        # Guard changes optimizer state even when no candidate fits. Restore
        # it on preflight budget stop so the last checkpoint stays coherent.
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        guard_state = {key: copy.deepcopy(getattr(self, key))
                       for key in self._state_fields()}
        event_count = len(self.ledger.events)
        self._guard()
        k = cfg["round_evals"]
        Q = cfg["proposal_multiplier"] * k
        proposals = self._propose(Q)
        keep = [i for i, (a, _, _) in enumerate(proposals) if not self._near_archive(a)]
        if not keep:
            keep = [int(self.rng.integers(len(proposals)))]
        proposals = [proposals[i] for i in keep]
        chosen_idx = list(self.rng.choice(len(proposals), size=min(k, len(proposals)),
                                          replace=False))
        chosen = []
        for i in chosen_idx:
            alpha, engine, meta = proposals[i]
            cid = self._register(alpha, engine, meta)
            chosen.append((cid, engine))
        # Preflight the entire round: an exhausted budget must never strand
        # partly evaluated candidates outside a completed-round checkpoint.
        alphas = [self.registry[cid]["alpha"] for cid, _ in chosen]
        full = sorted((self.oracle.estimate(a, None) for a in alphas), reverse=True)
        if self.ablation == "full_eval" or self.plateau_mode or (
                self.ablation != "no_plateau" and
                self.plateau_rounds >= cfg["plateau_patience"]):
            full_n = len(chosen) if self.ablation == "full_eval" else max(1, len(chosen) // 2)
            need = sum(full[:full_n])
        elif self.ablation == "standard_SH":
            m1 = min(len(chosen), max(cfg["sh_min_promote"],
                    math.ceil(len(chosen) * cfg["sh_eta"])))
            m2 = min(m1, max(cfg["sh_min_promote"], math.ceil(m1 * cfg["sh_eta"])))
            need = (sum(self.oracle.estimate(a, 1) for a in alphas)
                    + sum(sorted((self.oracle.estimate(a, 2) for a in alphas),
                                 reverse=True)[:m1]) + sum(full[:m2]))
        else:
            need = (sum(self.oracle.estimate(a, 1) for a in alphas)
                    + (0 if self.ablation == "no_rung2" else sum(
                        sorted((self.oracle.estimate(a, 2) for a in alphas),
                               reverse=True)))
                    + sum(full[:max(1, len(chosen) // 2)]))
        if need > self._budget_left() + 1e-9:
            del self.ledger.candidates[-len(chosen):]
            for cid, _ in chosen:
                del self.registry[cid]
            for key, value in guard_state.items():
                setattr(self, key, value)
            self.rng.bit_generator.state = rng_state
            del self.ledger.events[event_count:]
            raise BudgetExhausted("entire RACER round cannot fit remaining budget")
        promoted = self._funnel(chosen)
        self._after_promotion(promoted)
        self.sumdir_boost = False
        return True

    # ---------- 竞速漏斗 ----------
    def _funnel(self, chosen):
        """返回 [(cid, engine)]；按消融变体走不同路径。全部经 _call 计费。"""
        cfg = self.config
        ab = self.ablation
        if ab == "full_eval":
            return list(chosen)
        if ab == "standard_SH":
            return self._successive_halving(chosen)
        if self.plateau_mode or (ab != "no_plateau" and
                                 self.plateau_rounds >= cfg["plateau_patience"]):
            if not self.plateau_mode:
                self.ledger.events.append({"kind": "plateau_on", "candidate_id": None,
                                           "fidelity": None, "rung": None,
                                           "samples": 0.0, "score": None})
            self.plateau_mode = True
            self.plateau_rounds = 0
            kk = max(1, len(chosen) // 2)
            return list(chosen[:kk])
        coarse = {}
        for cid, _ in chosen:
            coarse[cid] = self._call(cid, 1)[0]
        rank_score = dict(coarse)
        if ab != "no_rung2" and len(chosen) > 1:
            best = max(coarse.values())
            near = [cid for cid, _ in chosen if best - coarse[cid] <= cfg["rung2_gap"]]
            if len(near) > 1:
                for cid in near:
                    rank_score[cid] = self._call(cid, 2)[0]
        order = sorted(chosen, key=lambda t: rank_score[t[0]], reverse=True)
        promote = order[:max(1, len(order) // 2)]
        self._plateau_stats(chosen, coarse)
        return promote

    def _successive_halving(self, chosen):
        cfg = self.config
        eta, floor_n = cfg["sh_eta"], cfg["sh_min_promote"]
        stage1 = list(chosen)
        rung1 = {cid: self._call(cid, 1) for cid, _ in stage1}
        order = sorted(stage1, key=lambda t: rung1[t[0]][0], reverse=True)
        m1 = max(floor_n, int(math.ceil(len(order) * eta)))
        stage2 = order[:min(m1, len(order))]
        if len(stage2) < len(stage1):
            rung2 = {cid: self._call(cid, 2) for cid, _ in stage2}
            order = sorted(stage2, key=lambda t: rung2[t[0]][0], reverse=True)
            m2 = max(floor_n, int(math.ceil(len(order) * eta)))
            return order[:min(m2, len(order))]
        return stage2

    def _plateau_stats(self, chosen, coarse):
        cfg = self.config
        if self.ablation in ("no_plateau",):
            return
        values = list(coarse.values())
        spread = float(max(values) - min(values)) if len(values) > 1 else 0.0
        rho = rankcorr(self.coarse_full_pairs[-6:])
        eps = cfg["plateau_eps_multitask"] if getattr(self.oracle, "is_multitask", False) \
            else cfg["plateau_eps_single"]
        if spread < eps and (rho is None or rho < cfg["plateau_rho"]):
            self.plateau_rounds += 1
        else:
            self.plateau_rounds = 0

    # ---------- 晋升后：DE 记忆 / anchor 更新 / evo_path ----------
    def _after_promotion(self, promoted):
        cfg = self.config
        med = float(np.median([self.registry[cid]["full"]
                               for cid in self.archive_ids])) \
            if self.archive_ids else -math.inf
        results = [(cid, engine, self._call(cid, None)) for cid, engine in promoted]
        succ_F, succ_CR = [], []
        for cid, engine, (score, _) in results:
            if engine == "de" and score > med:
                succ_F.append(self.registry[cid]["de_F"])
                succ_CR.append(self.registry[cid]["de_CR"])
            if self.ablation != "standard_SH" and self.ablation != "full_eval":
                for rung, rung_score in self.registry[cid]["coarse"].items():
                    self.coarse_full_pairs.append((rung_score, score))
                self.coarse_full_pairs = self.coarse_full_pairs[-12:]
        if succ_F:
            self.M_F[self.mem_ptr] = float(
                np.sum(np.array(succ_F) ** 2) / np.sum(succ_F))
            self.M_CR[self.mem_ptr] = float(np.mean(succ_CR))
            self.mem_ptr = (self.mem_ptr + 1) % len(self.M_F)
        best_cid, best_score = None, -math.inf
        for cid, _, (score, _) in results:
            if score > best_score:
                best_cid, best_score = cid, score
        if self.plateau_mode and best_score > self._anchor_score() + 1e-9:
            self.plateau_mode = False
            self.ledger.events.append({"kind": "plateau_off", "candidate_id": None,
                                       "fidelity": None, "rung": None,
                                       "samples": 0.0, "score": None})
        if best_score > self._anchor_score() + 1e-9:
            old = self._anchor_alpha()
            self.anchor_id = best_cid
            new = self._anchor_alpha()
            self.evo_path = (cfg["evo_decay"] * self.evo_path
                             + cfg["evo_gain"] * (new - old) / max(self.sigma, 1e-6))
            self.sigma = min(self.sigma * cfg["sigma_up"], cfg["sigma_max"])
            self.stagnation = 0
        else:
            self.stagnation += 1
            self.sigma = max(self.sigma * cfg["sigma_down"], cfg["sigma_min"])
            if self.stagnation >= cfg["stagnation_restart"]:
                self.sigma = min(self.sigma * cfg["sigma_restart"], cfg["sigma_max"])
                self.sumdir_boost = True
                self.ledger.events.append({"kind": "stagnation_restart", "candidate_id": None,
                                           "fidelity": None, "rung": None,
                                           "samples": 0.0, "score": None})
                self.stagnation = 0


class CmaFullSearch(_BaseSearch):
    """官方 pycma，完整同代 ask/tell，仅吃自身完整 dev 反馈。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="cma_full", ablation="full", **kwargs)
        import cma
        cfg = self.config["cma"]
        self._es = cma.CMAEvolutionStrategy(
            self.registry[self.best_seen_id]["alpha"].astype(float), cfg["sigma0"],
            {"popsize": cfg["popsize"], "verbose": -9,
             "bounds": [[self.config["lb"]] * self.n, [self.config["ub"]] * self.n],
             "seed": self.seed})
        self.generation = 0
        self._told_generations = []

    def _state_fields(self):
        return ("generation", "_told_generations")

    def _restore_optimizer(self):
        for generation in self._told_generations:
            asked = self._es.ask()
            if len(asked) != len(generation["xs"]) or any(
                    not np.allclose(a, b, rtol=0, atol=1e-10)
                    for a, b in zip(asked, generation["xs"])):
                raise ValueError("CMA replay ask mismatch")
            self._es.tell(asked, [-s for s in generation["scores"]])
        if len(self._told_generations) != self.generation:
            raise ValueError("CMA replay generation mismatch")

    def run(self, max_rounds=None):
        popsize = self.config["cma"]["popsize"]
        while max_rounds is None or self.generation < max_rounds:
            # CMA ask mutates internal optimizer state. Preflight a conservative
            # full-generation bound before ask so budget stops remain restorable.
            full_bound = self.oracle.estimate(self.registry[self.best_seen_id]["alpha"], None)
            if popsize * full_bound > self._budget_left() + 1e-9:
                self._checkpoint()
                return "budget_exhausted"
            asked = self._es.ask()
            if sum(self.oracle.estimate(self._clip(np.asarray(x, float)), None)
                   for x in asked) > self._budget_left() + 1e-9:
                raise RuntimeError("CMA full estimate varies by alpha; cannot safely checkpoint")
            batch = []
            for x in asked:
                cid = self._register(self._clip(np.asarray(x, float)), "cma")
                batch.append((cid, np.asarray(x, float)))
            self._at_boundary = False
            self._flush(batch)
            self.generation += 1
            self._checkpoint()
        self._checkpoint()
        return "max_rounds"

    def _flush(self, batch):
        if not batch:
            return
        scores = []
        for cid, _ in batch:
            score, _ = self._call(cid, None)
            scores.append(score)
        xs = [x for _, x in batch]
        self._es.tell(xs, [-s for s in scores])
        self._told_generations.append({"xs": [x.tolist() for x in xs], "scores": scores})


class CmaShSearch(_BaseSearch):
    """强组合对照：官方 CMA 在 rung1 粗目标上 ask/tell；每代固定比例经 rung2 晋升全评。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="cma_sh", ablation="full", **kwargs)
        import cma
        cfg = self.config["cma_sh"]
        self._es = cma.CMAEvolutionStrategy(
            self.registry[self.best_seen_id]["alpha"].astype(float), cfg["sigma0"],
            {"popsize": cfg["popsize"], "verbose": -9,
             "bounds": [[self.config["lb"]] * self.n, [self.config["ub"]] * self.n],
             "seed": self.seed})
        self.generation = 0
        self._told_generations = []

    def _state_fields(self):
        return ("generation", "_told_generations")

    def _restore_optimizer(self):
        for generation in self._told_generations:
            asked = self._es.ask()
            if len(asked) != len(generation["xs"]) or any(
                    not np.allclose(self._clip(a), b, rtol=0, atol=1e-10)
                    for a, b in zip(asked, generation["xs"])):
                raise ValueError("CMA-SH replay ask mismatch")
            self._es.tell([self._clip(np.asarray(x)) for x in asked],
                          [-s for s in generation["scores"]])
        if len(self._told_generations) != self.generation:
            raise ValueError("CMA-SH replay generation mismatch")

    def run(self, max_rounds=None):
        cfg = self.config["cma_sh"]
        while max_rounds is None or self.generation < max_rounds:
            # RealOracle estimates depend on rung, not alpha. Check full
            # generation before CMA ask mutates optimizer state.
            reference = self.registry[self.best_seen_id]["alpha"]
            bound = (cfg["popsize"] * self.oracle.estimate(reference, 1)
                     + cfg["rung2_promote"] * self.oracle.estimate(reference, 2)
                     + cfg["full_promote"] * self.oracle.estimate(reference, None))
            if bound > self._budget_left() + 1e-9:
                self._checkpoint()
                return "budget_exhausted"
            asked = [self._clip(np.asarray(x, float)) for x in self._es.ask()]
            # Reserve capacity for the entire generation before any rung is run.
            # Otherwise a rung2/full budget stop leaves CMA already told on a
            # generation that cannot be completed or resumed from this ledger.
            rung1 = sum(self.oracle.estimate(x, 1) for x in asked)
            rung2 = sorted((self.oracle.estimate(x, 2) for x in asked), reverse=True)
            full = sorted((self.oracle.estimate(x, None) for x in asked), reverse=True)
            need = (rung1 + sum(rung2[:cfg["rung2_promote"]])
                    + sum(full[:cfg["full_promote"]]))
            if need > self._budget_left() + 1e-9:
                raise RuntimeError("CMA-SH estimate varies by alpha; cannot safely checkpoint")
            self._at_boundary = False
            cids = [self._register(x, "cma") for x in asked]
            rung1 = {cid: self._call(cid, 1) for cid in cids}
            # CMA 只吃 rung1 粗分（真实观测，不伪造未全评候选的 full 值）
            scores = [rung1[cid][0] for cid in cids]
            self._es.tell([self.registry[cid]["alpha"] for cid in cids],
                          [-s for s in scores])
            self._told_generations.append({"xs": [x.tolist() for x in asked],
                                            "scores": scores})
            order = sorted(cids, key=lambda c: rung1[c][0], reverse=True)
            near = order[:min(cfg["rung2_promote"], len(order))]
            rung2 = {cid: self._call(cid, 2) for cid in near}
            finals = sorted(near, key=lambda c: rung2[c][0], reverse=True)
            for cid in finals[:min(cfg["full_promote"], len(finals))]:
                self._call(cid, None)
            self.generation += 1
            self._checkpoint()
        self._checkpoint()
        return "max_rounds"


class PsoBoxSearch(_BaseSearch):
    """search.py 公式移植到系数空间：显式同 box/同 init/同步更新、历史 worst 与 restart。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="pso_box", ablation="full", **kwargs)
        init_ids = list(self.archive_ids)
        if len(init_ids) != self.config["init_size"]:
            raise ValueError("PSO particles are the shared init points")
        self.positions = [self.registry[cid]["alpha"].copy() for cid in init_ids]
        self.velocities = []
        for i in range(len(init_ids)):
            j = int(self.rng.integers(len(init_ids)))
            while j == i:
                j = int(self.rng.integers(len(init_ids)))
            self.velocities.append(self.positions[j] - self.positions[i])  # random 起速
        self.p_best = [self.registry[cid]["full"] for cid in init_ids]
        self.p_best_x = [x.copy() for x in self.positions]
        self.g_best = max(self.p_best)
        self.g_worst = min(self.p_best)
        worst_idx = stable_argmax([-s for s in self.p_best])
        self._g_worst_x = self.positions[worst_idx].copy()
        self.histories = [[s] for s in self.p_best]
        self.step_length = self.config["pso"]["step_length"]
        self.iteration = 0
        self.g_history = [self.g_best]

    def _state_fields(self):
        return ("positions", "velocities", "p_best", "p_best_x", "g_best",
                "g_worst", "_g_worst_x", "histories", "step_length",
                "iteration", "g_history")

    def run(self, max_rounds=None):
        cfg = self.config["pso"]
        while max_rounds is None or self.iteration < max_rounds:
            if len(self.g_history) > cfg["patience"]:
                window = self.g_history[-cfg["patience"]:]
                if max(window) == min(window):
                    return "patience_stopped"
            results = self._iteration()
            if results is None:
                self._checkpoint()
                return "budget_exhausted"
            self.iteration += 1
            self._checkpoint()
        self._checkpoint()
        return "max_rounds"

    def _next_round_estimate(self):
        return sum(self.oracle.estimate(x, None) for x in self.positions)

    def _iteration(self):
        cfg = self.config["pso"]
        # 整代放不下则停止并报告剩余预算（不部分执行一代）
        if self._next_round_estimate() > self._budget_left() + 1e-9:
            return None
        self._at_boundary = False
        new_positions, new_velocities = [], []
        for i, x in enumerate(self.positions):
            history = self.histories[i]
            best_so_far = self.p_best[i]
            first_best = history.index(best_so_far)
            if len(history) - first_best >= cfg["restart_patience"] * cfg["patience"]:
                x = self.p_best_x[i].copy()      # stray restart：回到个人最优
                v = np.zeros(self.n)
            else:
                v = self.velocities[i]
            r_w, r_p, r_s, r_b = (float(self.rng.random()) for _ in range(4))
            w_self = r_w * cfg["inertia"]
            w_cog = r_p * cfg["cognitive"]
            w_soc = r_s * cfg["social"]
            w_rep = r_b * cfg["repel"]
            total = w_self + w_cog + w_soc + w_rep
            w_self, w_cog, w_soc, w_rep = (w / total for w in (w_self, w_cog, w_soc, w_rep))
            g_best_x = self.p_best_x[stable_argmax(self.p_best)]
            g_worst_x = self._g_worst_alpha()
            v_new = (w_self * v + w_cog * (self.p_best_x[i] - x)
                     + w_soc * (g_best_x - x) + w_rep * (x - g_worst_x))
            x_new = self._clip(x + self.step_length * v_new)
            new_positions.append(x_new)
            new_velocities.append(v_new)
        self.positions, self.velocities = new_positions, new_velocities
        for i, x in enumerate(self.positions):
            cid = self._register(x, "pso")
            score, _ = self._call(cid, None)
            self.histories[i].append(score)
            if score > self.p_best[i]:
                self.p_best[i] = score
                self.p_best_x[i] = x.copy()
            if score > self.g_best:
                self.g_best = score
            if score < self.g_worst:
                self.g_worst = score
                self._g_worst_x = x.copy()
        self.g_history.append(self.g_best)
        self.step_length = max(self.step_length * cfg["step_factor"], cfg["min_step"])
        return True

    def _particle_score(self, j):
        return self.histories[j][-1]

    def _g_worst_alpha(self):
        return getattr(self, "_g_worst_x", self.positions[0].copy())


class PsoSum1Search(_BaseSearch):
    """C 组机制对照：符号 sum1 空间 PSO（box [lb,ub] 内、Σα=1 的仿射壳）。

    scalegrid 变体：同一 PSO 轨迹，但预先保留固定 scale 网格
    [.5,.75,1,1.25,1.5] 的全评额度（每次=完整 dev），在 PSO 结束后
    对其最优若干候选做网格评估；无 scale 版把对应预算全部用于搜索。
    """

    def __init__(self, *args, scale_grid=None, scale_candidates=2, **kwargs):
        super().__init__(*args, method="pso_sum1" + ("_scalegrid" if scale_grid else ""),
                         ablation="full", **kwargs)
        self.scale_grid = list(scale_grid) if scale_grid else None
        self.scale_candidates = scale_candidates
        init_ids = list(self.archive_ids)
        self.positions = [self._project(self.registry[cid]["alpha"]) for cid in init_ids]
        self.velocities = []
        for i in range(len(init_ids)):
            j = int(self.rng.integers(len(init_ids)))
            while j == i:
                j = int(self.rng.integers(len(init_ids)))
            self.velocities.append(self.positions[j] - self.positions[i])
        self.p_best = [self.registry[cid]["full"] for cid in init_ids]
        self.p_best_x = [x.copy() for x in self.positions]
        self.g_history = [max(self.p_best)]
        self.step_length = self.config["pso"]["step_length"]
        self.iteration = 0
        self._grid_done = False

    def _state_fields(self):
        return ("positions", "velocities", "p_best", "p_best_x", "g_history",
                "step_length", "iteration", "_grid_done")

    def _project(self, a):
        """欧氏投影到 box ∩ {Σα=1}（Held 界 + 二分求偏移 τ，精确到 1e-12）。

        解 min ||a-τ·1||² s.t. lb ≤ a ≤ ub；二分 τ 使 Σclip(aᵢ,lb,ub)=1。
        box [−.5,1.5]²⁰ 与 Σ=1 有非空交集（如 onehot），可行域保证收敛。
        """
        a = np.asarray(a, dtype=np.float64)
        lb, ub = self.config["lb"], self.config["ub"]
        target = 1.0

        def total(tau):
            return float(np.clip(a - tau, lb, ub).sum())

        lo, hi = float(a.min() - ub), float(a.max() - lb)  # total(hi)<=target<=total(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if total(mid) > target:
                lo = mid
            else:
                hi = mid
        return np.clip(a - 0.5 * (lo + hi), lb, ub)

    def _search_budget(self):
        """scale 版预留网格额度给搜索的净预算。"""
        if not self.scale_grid:
            return self.budget_samples
        # The frozen oracle's full-dev size is independent of alpha. Keep this
        # explicit: an alpha-dependent estimate cannot safely reserve grid cost.
        reference = self.oracle.estimate(self.registry[self.best_seen_id]["alpha"], None)
        reserve = len(self.scale_grid) * self.scale_candidates * reference
        return self.budget_samples - reserve

    def run(self, max_rounds=None):
        cfg = self.config["pso"]
        search_budget = self._search_budget()
        while (not self._grid_done and
               (max_rounds is None or self.iteration < max_rounds)):
            if len(self.g_history) > cfg["patience"]:
                window = self.g_history[-cfg["patience"]:]
                if max(window) == min(window):
                    break
            if sum(self.oracle.estimate(x, None) for x in self.positions) > \
                    self._budget_left_search(search_budget) + 1e-9:
                break
            self._at_boundary = False
            self._iteration()
            self.iteration += 1
            self._checkpoint()
        if self.scale_grid and not self._grid_done:
            self._at_boundary = False
            self._scale_grid_phase()
            self._grid_done = True
        self._checkpoint()
        return "done"

    def _budget_left_search(self, search_budget):
        return search_budget - self.ledger.budget_used

    def _iteration(self):
        cfg = self.config["pso"]
        g_best_x = self.p_best_x[stable_argmax(self.p_best)]
        g_worst_x = self.p_best_x[stable_argmax([-s for s in self.p_best])]
        new_pos, new_vel = [], []
        for i, x in enumerate(self.positions):
            r_w, r_p, r_s, r_b = (float(self.rng.random()) for _ in range(4))
            w_self = r_w * cfg["inertia"]
            w_cog = r_p * cfg["cognitive"]
            w_soc = r_s * cfg["social"]
            w_rep = r_b * cfg["repel"]
            total = w_self + w_cog + w_soc + w_rep
            w_self, w_cog, w_soc, w_rep = (w / total
                                           for w in (w_self, w_cog, w_soc, w_rep))
            v = (w_self * self.velocities[i] + w_cog * (self.p_best_x[i] - x)
                 + w_soc * (g_best_x - x) + w_rep * (x - g_worst_x))
            x_new = self._project(x + self.step_length * v)
            new_pos.append(x_new)
            new_vel.append(v)
        self.positions, self.velocities = new_pos, new_vel
        for i, x in enumerate(self.positions):
            cid = self._register(x, "pso_sum1")
            score, _ = self._call(cid, None)
            if score > self.p_best[i]:
                self.p_best[i] = score
                self.p_best_x[i] = x.copy()
        self.g_history.append(max(self.p_best))
        self.step_length = max(self.step_length * cfg["step_factor"], cfg["min_step"])

    def _scale_grid_phase(self):
        """对 top-K 候选 × 固定 scale 网格全评（box 满足：grid 值仍在 [lb,ub]×系数域）。"""
        order = sorted(self.archive_ids,
                       key=lambda c: -self.registry[c]["full"])
        for cid in order[:self.scale_candidates]:
            base = self.registry[cid]["alpha"]
            base_sum = float(base.sum())
            if abs(base_sum) < 1e-9:
                continue
            for scale in self.scale_grid:
                scaled = self._clip(base * (scale / max(abs(base_sum), 1e-9) * (1 if base_sum > 0 else -1)))
                scid = self._register(scaled, "scale_grid")
                try:
                    self._call(scid, None)
                except BudgetExhausted:
                    return


class RandomBoxSearch(_BaseSearch):
    """相同 signed box 均匀采样（非负 Dirichlet 旧基线不沿用）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="random_box", ablation="full", **kwargs)
        self.round_index = 0
        self.batch_progress = 0

    def _state_fields(self):
        return ("round_index", "batch_progress")

    def run(self, max_rounds=None):
        cfg = self.config
        seen = [self.registry[cid]["alpha"] for cid in self.archive_ids]
        while max_rounds is None or self.round_index < max_rounds:
            for _ in range(self.batch_progress, cfg["random_batch"]):
                rng_before = copy.deepcopy(self.rng.bit_generator.state)
                a = self._sample_box()
                tries = 0
                while any(np.linalg.norm(a - b) < cfg["dedup_eps"] for b in seen) \
                        and tries < cfg["random_dedup_tries"]:
                    a = self._sample_box()
                    tries += 1
                # Check the actual candidate before registering or spending samples.
                if self.oracle.estimate(a, None) > self._budget_left() + 1e-9:
                    self.rng.bit_generator.state = rng_before
                    self._checkpoint()
                    return "budget_exhausted"
                self._at_boundary = False
                cid = self._register(a, "random")
                self._call(cid, None)
                seen.append(a)
                self.batch_progress += 1
                self._checkpoint()
            self.round_index += 1
            self.batch_progress = 0
            self._checkpoint()
        self._checkpoint()
        return "max_rounds"

    def _sample_box(self):
        return self.rng.uniform(self.config["lb"], self.config["ub"], self.n)


def build_search(method, ablation, n, config, oracle, rng, init_alphas,
                 init_scores, init_costs, budget_samples, seed=42):
    if method == "racer_frozen_v1":
        return RacerSearch(n, config, oracle, rng, init_alphas, init_scores,
                           init_costs, budget_samples, ablation=ablation, seed=seed)
    if method == "cma_full":
        return CmaFullSearch(n, config, oracle, rng, init_alphas, init_scores,
                             init_costs, budget_samples, seed=seed)
    if method == "cma_sh":
        return CmaShSearch(n, config, oracle, rng, init_alphas, init_scores,
                           init_costs, budget_samples, seed=seed)
    if method == "pso_box":
        return PsoBoxSearch(n, config, oracle, rng, init_alphas, init_scores,
                            init_costs, budget_samples, seed=seed)
    if method == "random_box":
        return RandomBoxSearch(n, config, oracle, rng, init_alphas, init_scores,
                               init_costs, budget_samples, seed=seed)
    if method == "pso_sum1":
        return PsoSum1Search(n, config, oracle, rng, init_alphas, init_scores,
                             init_costs, budget_samples, seed=seed)
    if method == "pso_sum1_scalegrid":
        return PsoSum1Search(n, config, oracle, rng, init_alphas, init_scores,
                             init_costs, budget_samples, seed=seed,
                             scale_grid=[0.5, 0.75, 1.0, 1.25, 1.5])
    raise ValueError("unknown method: " + str(method))
