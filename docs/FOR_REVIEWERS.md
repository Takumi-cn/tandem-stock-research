# For Reviewers / 给评审的 60 秒说明

## 60-second summary

This repository is a **falsification-first quantitative research workstation**, not a pitch for a profitable strategy.

The author built a complete A-share research system:

- data ingestion and feature contracts,
- a compact ensemble (LightGBM + deep models + L0 factors + zero-shot Kronos),
- a 55% no-trade gate,
- T+1 prediction ledger and next-day review,
- a native PySide6 desktop UI,
- leakage / PIT / cost / multiple-comparison audits.

Then the system was used to test whether the compact model survives realistic costs. It did not:

- 742 OOS days, Top1 account: 1.011x at 14.5 bps/side, **0.463x at 25 bps**, 0.151x at 40 bps.
- Maximum cohort drawdown: **−80.2%**.
- 120-day realizable 2-day net at 14.5 bps: **+0.043%**, 95% CI [−1.153%, +1.302%], p(≤0)=0.477.
- Several model/industry branches were rejected and are documented.

The unusual and, I believe, more valuable contribution is the **honest negative result**. The repository demonstrates the ability to:

1. formulate a falsifiable question;
2. build the data and ML system needed to test it;
3. prevent leakage and overfit through contracts, purged folds and cost stress;
4. stop when the evidence says stop;
5. communicate the result in bilingual documentation.

## Why this is relevant to research engineering / data roles

- **Systems**: end-to-end pipeline, native desktop UI, async feed, cache fallback, failure recovery, resource bounds.
- **ML engineering**: ensembling, state routing, probability ranking, calibration and OOS diagnostics — not a single notebook metric.
- **Data engineering**: multi-source ingestion, feature availability contracts, stale-data guards, deterministic tests.
- **Research integrity**: pre-registration, matched baselines, effective sample size, DSR/PBO, confidence intervals, minimum detectable effect.
- **Communication**: explicit non-claims, reproducible commands, English/Chinese docs.

## What this does *not* claim

- It does not claim a profitable after-cost strategy.
- It does not claim a validated real-time execution edge.
- It does not include paid data, broker connectivity, or order routing.
- It does not use the UI screenshot as investment advice.

## Where to look first

| Path | Why |
|---|---|
| `docs/RESEARCH_STATUS.md` | Evidence, failed branches, cost/CI tables |
| `src/eval/top1_audit.py` | Tradability, costs, bootstrap and gate audit |
| `src/eval/metric_reconciliation.py` | Why opportunity/per-trade/account returns must not be mixed |
| `tests/test_no_lookahead.py` | Leakage guards |
| `tests/test_contract_v2.py` | Feature/label availability contract |
| `src/gui/desktop.py` | Native PySide6 product implementation |
| `src/gui/industry_feed.py` | Async industry data feed with cache/failure handling |
| `docs/07_执行手册.md` | Reproducible operating procedure |

## Reproduce the core checks

```bash
python -m venv .venv
pip install -r requirements.txt
pytest tests
```

Market data and model weights are not redistributed. The README explains how to obtain/rebuild them.

## 中文摘要

这是一个“证伪优先”的量化研究工程项目，而不是盈利宣传。作者搭建了数据、特征、模型集成、55% 风控门槛、T+1 台账、复盘和原生桌面界面，然后用严格的成本/泄漏/多重比较审计去检验它。结论是：**没有证明成本后优势**；742 天样本外在 25bps/边下账户为 0.463x，120 天可实现净收益置信区间跨 0。项目保留这些负面结果，并记录了失败研究方向。

如果作为研究工程/数据工程作品评审，建议重点看：端到端系统能力、PIT/泄漏防护、成本与统计功效意识、以及明确写下“不声称赚钱”的沟通诚实度。
