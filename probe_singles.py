"""probe_singles.py — 可组合性判据补数: 10 个单专家 one-hot 全评(两任务)."""
import json, logging, os, socket, sys
import numpy as np
from alpha_space import ExpertPool
from evaluate import evaluate
from search_lp import eval_batch


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
    ET = "exact_match"
    logging.basicConfig(filename=f"probe_singles_{task}.log", level=logging.INFO,
                        format="%(asctime)s %(message)s")
    pool = ExpertPool("./initial_experts")
    n = pool.n
    dirs = [pool.materialize(np.eye(n)[i], f"single_{i}") for i in range(n)]
    scores = eval_batch(dirs, ET, task, [0, 1, 2, 3], "Qwen/Qwen2.5-1.5B-Instruct")
    c_scores = eval_batch(dirs, ET, task, [0, 1, 2, 3], "Qwen/Qwen2.5-1.5B-Instruct", subset_size=50)
    for i, (nm, s) in enumerate(zip(pool.names, scores)):
        logging.info(f"single {i:2d} {nm:<20} {s:.3f}")
    best = max(scores)
    logging.info(f"BEST_SINGLE {task}: {best:.3f} ({pool.names[int(np.argmax(scores))]})")
    logging.info(f"BEST_SINGLE_COARSE {task}: {max(c_scores):.3f}")
    print(f"{task} best single = {best:.3f}")

