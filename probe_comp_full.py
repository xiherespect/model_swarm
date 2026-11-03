"""probe_comp_full.py — 全保真版可组合性判据(4代表任务: 用dev全量200条)."""
import json, logging
import numpy as np
from alpha_space import ExpertPool
from search_lp import eval_batch

logging.basicConfig(filename="probe_comp_full.log", level=logging.INFO,
                    format="%(asctime)s %(message)s")

TASKS = [("nlgraph", "exact_match"), ("gsm8k", "exact_match"),
         ("science", "multitask"), ("legal", "multitask")]

if __name__ == "__main__":
    pool = ExpertPool("./initial_experts")
    n = pool.n
    results = {}
    for ds, et in TASKS:
        logging.info(f"===== {ds} =====")
        eye = [pool.materialize(np.eye(n)[i], f"full_s_{ds}_{i}") for i in range(n)]
        singles = eval_batch(eye, et, ds, [0,1,2,3], "Qwen/Qwen2.5-1.5B-Instruct")
        bs = float(max(singles))
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        pm = []
        for i, j in pairs:
            a = np.zeros(n); a[i] = a[j] = 0.5
            pm.append(pool.materialize(a, f"full_p_{ds}_{i}_{j}"))
        pc = eval_batch(pm, et, ds, [0,1,2,3], "Qwen/Qwen2.5-1.5B-Instruct")
        bp = float(max(pc))
        ratio = bp / max(bs, 1e-6)
        results[ds] = {"best_single": bs, "best_pair": bp, "ratio": ratio,
                       "composable": ratio >= 0.90}
        logging.info(f"{ds}: single={bs:.3f} pair={bp:.3f} ratio={ratio:.3f} "
                     f"{'COMPOSABLE' if ratio >= 0.90 else 'NOT'}")
    with open("composability_full_results.json", "w") as f:
        json.dump(results, f, indent=1)
    for ds, r in results.items():
        print(f"{ds:12s} ratio={r['ratio']:.3f} {'COMPOSABLE' if r['composable'] else 'NOT'}")
