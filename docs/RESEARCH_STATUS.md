# Research status / 研究状态

> Last updated: 2026-10-29. This document records evidence, including failed experiments. It is not investment advice.

## Bottom line

The production pipeline is a research prototype. **No after-cost alpha has been demonstrated.** The 55% gate is a risk-control rule; it is not a validated profit guarantee. Failed directions are not restarted and production weights are not changed without a pre-registered, out-of-sample test.

## Return and cost evidence

### Long-window Top1 (19 purged folds, 742 OOS days, 2023-09..2026-10)

| Cost assumption | Account path |
|---|---:|
| 14.5 bps/side | 1.011x |
| 25 bps/side | 0.463x |
| 40 bps/side | 0.151x |

- Max cohort drawdown: **-80.2%**
- Daily volatility ~6.4%; daily turnover ~94%
- Effective independent observations ≈371 (2-day overlapping labels); minimum detectable mean daily net return ≈0.829%/day
- Arithmetic net mean can look positive while geometric/account return is flat or negative. Do not quote arithmetic mean as annual return.

### TOP1-AUDIT-002 (120 signal days, 2026-04-27..2026-10-23, T+1 open→T+2 close)

- B1 per-trade mean: +0.379% over 79 trades; median -0.108%; win rate 46.8% (mean driven by a few large winners)
- B1 daily opportunity mean: +0.250% over all 120 days, including 41 no-trade days; not an account return
- B1-only account: dual-sleeve 1.1294x / single-sleeve 0.6947x (execution and overlap assumptions matter)
- All-Top1 account: 0.9566x (different definition; do not mix)
- Realizable 2-day net at 14.5 bps: +0.043%, 95% CI [-1.153%, +1.302%], p(≤0)=0.477
- 25 bps stress: negative
- Gated days net: +0.379%, 95% CI [-1.265%, +2.036%], p=0.329

## Closed / failed research branches

- `MICRO-005-ENC-002`: compact encoder showed no incremental value; decision `ENCODER_FAIL`
- Industry-layer v4 / v4.1 / Experiment A: rejected; no stock-layer promotion
- Page top-K / portfolio expansion: not better than Top1 under the tested definitions
- Breadth: feasibility passes for historical PIT subset but exact limit price is only a proxy and realtime breadth is unavailable; retained as a 60-day shadow only
- Calibration/conformal and reliability work: useful diagnostics, no production promotion without forward evidence

## Data constraints

- Free daily sources are not fully point-in-time; current industry classification is not historical PIT
- Exact limit-up/down price and realtime breadth are unavailable; only proxies are used
- Delisted-stock coverage and suspension handling remain limited
- Kronos weights and market data are not redistributed

## Decision rule

Do not increase position size or promote a new model based on historical replay. A candidate must pass a pre-registered forward test with costs, matched-random/permutation baselines, and a minimum sample; otherwise it stays research-only. The current production default is to respect the 55% gate and not trade when the probability is below it.
