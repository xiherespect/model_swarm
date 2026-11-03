# Why this gate cohort was superseded (2026-10-23)

The driver `tools/run_racer_campaign.py` and its test file gained the
`--resume` feature (campaign ledger append-mode, needed for phase-3 formal
matrix scheduling). Both files are identity dependencies hashed into the
campaign snapshot, so every snapshot in this cohort no longer verifies
against current sources (`verify_campaign_identity` -> source mismatch).

Per the frozen plan ("基础设施错误只在身份不变…恢复；需要改代码则新版本整
cohort 重跑，不拼接旧版成功结果"), the successful runs here (G1 nlgraph
94.7s; G2 nlgraph racer 820.5s / cma 819.2s / pso 823.3s / random 1014.2s;
G2 gsm8k racer 1878.4s) are NOT stitched into the replacement gate. They
remain as debugging evidence: throughput ~5.9 samples/s, gsm8k generation
~2.3x slower, budget ledger semantics verified.

Replacement cohort: experiments/racer_gate_v1 (fresh preflight, all 25 runs).
