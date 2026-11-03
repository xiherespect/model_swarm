# D_kcross 重跑 cohort（修复后身份）— 披露说明

背景：BD 实例的 D_kcross 12 run 因 `search_frozen.evaluate_test` 把任务名
`kcross` 当数据集名传给 `evaluate.evaluate_test` 全部失败（实际文件
`data/eval/knowledge_crosswords.json`；dev 搜索不受影响，仅 test 一步）。
用户 2026-10-28 决定：修复 + 只重跑 D 组 12 run，A/B/C 228 run 旧身份保留。

- 修复：search_frozen.py evaluate_test 单数据集任务改传 self.datasets[0]，
  multitask 不变。回归锁：tests/test_search_frozen_runner.py kcross 用例
  断言传出 dataset 参数为 knowledge_crosswords（8/8 runner 测试通过）。
- 本 cohort 身份 ≠ AC/BD 身份（search_frozen.py 是身份依赖）。论文呈现时
  明确披露该拼接；D 组单独一节敏感性分析，不与 A/B/C 混算宏平均（本来也不混）。
- 排程约束：工具冻结 max_parallel≤2、GPU 白名单≤2 张，故拆两实例
  D1（s42 ×4 run，GPU 0,1）+ D2（s43+s44 ×8 run，GPU 2,3），不改代码。
- BD 账本 3 个 failed + 2 个 cancelled 留档不删；旧 D run 目录待归档。

## D 组终态（2026-10-29 收官）

D1 4/4 + D2 8/8 全部 succeeded，双实例 ledger 均 succeeded，四卡释放。

| 方法 | test 均值 | filtered 均值 | Δ(filtered−test) |
|---|---|---|---|
| racer_frozen_v1 | .3980 | .3989 | +0.1pt |
| cma_full | .4147 | .4285 | +1.4pt |
| pso_box | .3940 | .4001 | +0.6pt |
| random_box | .3930 | .3989 | +0.6pt |

要点：cma 在 kcross 上领先且 filtered 增益最大；racer filtered≈test（污染不敏感）；
该任务 racer 搜索负增益（落后 cma ~1.7pt）与 truthfulqa 同类画像，如实呈现。
