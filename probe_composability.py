"""probe_composability.py — 可组合性判据全谱: ratio = best_pair / best_single (同粗评保真)."""
import json, logging, sys
import numpy as np
from alpha_space import ExpertPool
from search_lp import eval_batch

logging.basicConfig(filename="probe_composability.log", level=logging.INFO,
                    format="%(asctime)s %(message)s")

TASKS = [
    ("nlgraph", "exact_match", "nlgraph"), ("gsm8k", "exact_match", "gsm8k"),
    ("hellaswag", "multiple_choice", "hellaswag"), ("legal", "multitask", "legal"),
    ("medical", "multitask", "medical"), ("culture", "multitask", "culture"),
    ("science", "multitask", "science"), ("kcross", "multiple_choice", "knowledge_crosswords"),
    ("mmlu", "multiple_choice"), ("mmlu_pro", "multiple_choice"),
    ("truthfulqa", "multiple_choice"), ("abstainqa", "AbstainQA", "mmlu"),
]


if __name__ == "__main__":
    pool = ExpertPool("./initial_experts")
    n = pool.n
    results = {}

    for nm, et, ds in TASKS:
        logging.info(f"===== {ds} =====")
        # 10 单专家粗评
        eye_dirs = [pool.materialize(np.eye(n)[i], f"cmp_single_{ds}_{i}")
                    for i in range(n)]
        singles = eval_batch(eye_dirs, et, ds, [0,1,2,3], "Qwen/Qwen2.5-1.5B-Instruct",
                             subset_size=50)
        best_single = float(max(singles))

        # 45 对中点粗评
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        pm_dirs = []
        for i, j in pairs:
            a = np.zeros(n); a[i] = a[j] = 0.5
            pm_dirs.append(pool.materialize(a, f"cmp_pair_{ds}_{i}_{j}"))
        pair_c = eval_batch(pm_dirs, et, ds, [0,1,2,3], "Qwen/Qwen2.5-1.5B-Instruct",
                            subset_size=50)
        best_pair = float(max(pair_c))
        best_pair_idx = int(np.argmax(pair_c))

        ratio = best_pair / max(best_single, 1e-6)
        results[ds] = {"best_single": best_single, "best_pair": best_pair,
                       "ratio": ratio, "composable": ratio >= 0.95,
                       "best_pair_experts": list(pairs[best_pair_idx])}
        logging.info(f"{ds}: single={best_single:.3f} pair={best_pair:.3f} "
                     f"ratio={ratio:.3f} {'COMPOSABLE' if ratio >= 0.95 else 'NOT'} "
                     f"pair={pairs[best_pair_idx]}")

    # 保存
    with open("composability_results.json", "w") as f:
        json.dump(results, f, indent=1)
    logging.info("ALL DONE")
    for ds, r in sorted(results.items(), key=lambda x: x[1]["ratio"], reverse=True):
        print(f"{ds:12s} ratio={r['ratio']:.3f} {'✓' if r['composable'] else '✗'} "
              f"single={r['best_single']:.3f} pair={r['best_pair']:.3f}")
