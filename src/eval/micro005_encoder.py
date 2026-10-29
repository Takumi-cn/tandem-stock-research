"""Compact 5-minute encoder for TANDEM-FWD-MICRO-005 (shadow only, gated).

Gated by the pre-registered minimal test: this module refuses to run unless
`reports/micro005_minimal/decision.json` says `INTRADAY_CANDIDATE_SHADOW_ONLY`
(or `--force-diagnostic` is passed explicitly, which marks the run as a smoke
test and forbids reporting it as evidence).

Architecture (deliberately ~0.1M parameters, per the STRATA lesson that richer
raw information matters more than model size):

    5 x 48 x 8  ->  causal depthwise conv k=3/6/12/24  ->  concat 32
                ->  pointwise 64  ->  GRU(64, 2 layers)
                ->  attention pool + mean pool  ->  32-d embedding
                ->  head with the same daily whitelist as arm A/B

Evaluation uses the *same* folds and target as the minimal test, so the three
arms are directly comparable.  Even a win here is `ENCODER_DIAGNOSTIC_ONLY`:
the 2021-2026 period cannot certify anything, and production stays frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.feature_contract_v2 import DAILY_ARM, DAILY_FEATURES, MARKET_FEATURES, TARGET, select_X  # noqa: E402
from src.eval.micro005_minimal import (  # noqa: E402
    DATA, OUT, PURGE_DAYS, SEEDS, boot_p, daily_metrics, folds, log,
)
from src.utils.config import ROOT  # noqa: E402

CACHE5 = ROOT / "data" / "raw" / "5m"
ENC_OUT = OUT / "encoder"
ENC002_OUT = ROOT / "reports" / "micro005_encoder_002"
ENC002_DATA = ENC002_OUT / "data"
ENC002_BASE = ROOT / "data" / "research" / "micro005_encoder_002"
N_DAYS, N_BARS, N_CH = 5, 48, 8
MAX_PARAMS = 300_000


def build_tensors(codes: list[str] | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """(5, 48, 8) tensors for every page row that has 5 prior sessions of 5m bars."""
    page = pd.read_parquet(DATA / "page_dataset.parquet")
    page["date"] = pd.to_datetime(page["date"]); page["code"] = page["code"].astype(str)
    page = page.sort_values(["date", "code"]).reset_index(drop=True)
    by_code = {c: g.index.to_numpy() for c, g in page.groupby("code")}
    slots = pd.date_range("2026-01-01 09:35", "2026-01-01 11:30", freq="5min").tolist() + \
        pd.date_range("2026-01-01 13:05", "2026-01-01 15:00", freq="5min").tolist()
    slot_map = {t.hour * 100 + t.minute: i for i, t in enumerate(slots)}
    arr = np.zeros((len(page), N_DAYS, N_BARS, N_CH), dtype=np.float16)
    have = np.zeros(len(page), dtype=bool)
    codes = codes or sorted(by_code)
    for k, code in enumerate(codes, 1):
        p = CACHE5 / f"{code}.parquet"
        if not p.exists():
            continue
        raw = pd.read_parquet(p)
        raw["date"] = pd.to_datetime(raw["date"])
        raw["slot"] = raw["hhmm"].astype(int).map(slot_map)
        raw = raw.dropna(subset=["slot"])
        days = pd.DatetimeIndex(sorted(raw["date"].unique()))
        if len(days) < N_DAYS + 2:
            continue
        piv = {c: raw.pivot(index="date", columns="slot", values=c)
               .reindex(index=days, columns=range(N_BARS)).to_numpy(dtype=np.float32)
               for c in ("open", "high", "low", "close", "volume", "amount")}
        O, H, L, C, V, A = (piv[c] for c in ("open", "high", "low", "close", "volume", "amount"))
        with np.errstate(invalid="ignore", divide="ignore"):
            prev_bar = np.concatenate([O[:, :1], C[:, :-1]], axis=1)
            R = C / np.where(prev_bar > 0, prev_bar, np.nan) - 1.0
            RNG = (H - L) / np.where(O[:, :1] > 0, O[:, :1], np.nan)
            CP = (C - L) / np.where((H - L) > 0, H - L, np.nan)
            VZbase = pd.DataFrame(V).rolling(20, min_periods=5).mean().to_numpy()
            VZ = V / np.where(VZbase > 0, VZbase, np.nan) - 1.0
            ASH = A / np.where(np.nansum(A, axis=1, keepdims=True) > 0,
                               np.nansum(A, axis=1, keepdims=True), np.nan)
        R = np.clip(np.nan_to_num(R), -0.1, 0.1)
        RNG = np.clip(np.nan_to_num(RNG), 0, 0.2)
        CP = np.nan_to_num(CP, nan=0.5)
        VZ = np.clip(np.nan_to_num(VZ), -5, 5) / 5.0
        ASH = np.nan_to_num(ASH)
        day_sigma = np.maximum(np.nanstd(R, axis=1, keepdims=True), 1e-4)
        Rn = np.clip(R / day_sigma, -4, 4) / 4.0
        sin_t = np.sin(2 * np.pi * np.arange(N_BARS) / N_BARS)[None, :]
        cos_t = np.cos(2 * np.pi * np.arange(N_BARS) / N_BARS)[None, :]
        for idx in by_code.get(code, []):
            d = page.at[idx, "date"]
            j = days.get_loc(d) if d in days else -1
            if j < N_DAYS - 1:
                continue
            sl = slice(j - N_DAYS + 1, j + 1)
            t = np.empty((N_DAYS, N_BARS, N_CH), dtype=np.float32)
            t[..., 0] = R[sl]
            t[..., 1] = Rn[sl]
            t[..., 2] = VZ[sl]
            t[..., 3] = RNG[sl] * 5
            t[..., 4] = (CP[sl] - 0.5) * 2
            t[..., 5] = np.log1p(ASH[sl] * 100) / 5
            t[..., 6] = sin_t
            t[..., 7] = cos_t
            arr[idx] = t.astype(np.float16)
            have[idx] = True
        if k % 20 == 0:
            log(f"  tensors {k}/{len(codes)} covered={int(have.sum()):,}")
    meta = page[["date", "code", "rank_in_fp", TARGET, "label__exec_T1_T2"]].copy()
    meta["has_tensor"] = have
    return arr, meta


def _make_model(daily_dim: int):
    import torch
    import torch.nn as nn

    class CausalDW(nn.Module):
        def __init__(self, ch, k):
            super().__init__()
            self.k = k
            self.conv = nn.Conv1d(ch, ch, k, padding=0, groups=ch)

        def forward(self, x):  # x: (B, ch, T)
            return self.conv(nn.functional.pad(x, (self.k - 1, 0)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.dw = nn.ModuleList([CausalDW(N_CH, k) for k in (3, 6, 12, 24)])
            self.pw = nn.Conv1d(N_CH * 4, 64, 1)
            self.gru = nn.GRU(64, 64, num_layers=2, batch_first=True, dropout=0.1)
            self.att = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
            self.emb = nn.Sequential(nn.Linear(128, 32), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(32 + daily_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(64, 1))

        def forward(self, x, daily):
            # x: (B, days, bars, ch) -> (B, ch, days*bars)
            x = x.permute(0, 3, 1, 2).flatten(2)
            z = torch.cat([m(x) for m in self.dw], dim=1)
            z = torch.relu(self.pw(z)).transpose(1, 2)
            h, _ = self.gru(z)
            a = torch.softmax(self.att(h).squeeze(-1), dim=1).unsqueeze(-1)
            pooled = (h * a).sum(1)
            emb = self.emb(torch.cat([pooled, h.mean(1)], dim=1))
            return self.head(torch.cat([emb, daily], dim=1)).squeeze(-1), emb

    return Net()


def _prep_daily(Dtr: np.ndarray, Dte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold-safe imputation + standardisation. Trees tolerate NaN; the net does not.

    Medians/means/stds are computed on the training block only, so the test block
    never informs preprocessing. Without this the 1.2% NaN daily features turn the
    first MSE step into NaN and every seed emits NaN predictions.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        med = np.nanmedian(Dtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Dtr = np.where(np.isfinite(Dtr), Dtr, med)
    Dte = np.where(np.isfinite(Dte), Dte, med)
    mu = Dtr.mean(axis=0)
    sd = Dtr.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Dtr = np.clip((Dtr - mu) / sd, -10.0, 10.0)
    Dte = np.clip((Dte - mu) / sd, -10.0, 10.0)
    return Dtr.astype(np.float32), Dte.astype(np.float32)


def _check_pred(p: np.ndarray, tag: str) -> None:
    if not np.isfinite(p).all():
        raise RuntimeError(f"non-finite predictions in {tag}")
    if float(np.std(p)) < 1e-12:
        raise RuntimeError(f"constant predictions in {tag}")


def _ckpt_ok(path: Path) -> bool:
    try:
        df = pd.read_parquet(path, columns=["pred"])
        p = df["pred"].to_numpy(dtype=float)
    except Exception:
        return False
    return len(p) > 0 and bool(np.isfinite(p).all()) and float(np.std(p)) > 1e-12


def train_fold(Xtr, Dtr, ytr, Xte, Dte, seed: int, epochs: int = 18, bs: int = 256):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _make_model(Dtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    n = len(Xtr)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    # Whole fold resident on GPU: the old loop copied every 256-row batch over
    # PCIe, which dominated the ~92 s/seed runtime for a 63k-parameter model.
    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr)).to(dev)
    Dtr_t = torch.from_numpy(Dtr).to(dev)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    for ep in range(epochs):
        rng.shuffle(idx)
        model.train()
        for s in range(0, n, bs):
            b = torch.from_numpy(idx[s:s + bs]).to(dev)
            opt.zero_grad()
            pred, _ = model(Xtr_t[b], Dtr_t[b])
            loss = lossf(pred, ytr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
    model.eval()
    Xte_t = torch.from_numpy(np.ascontiguousarray(Xte)).to(dev)
    Dte_t = torch.from_numpy(Dte).to(dev)
    outs = []
    with torch.no_grad():
        for s in range(0, len(Xte), 512):
            outs.append(model(Xte_t[s:s + 512], Dte_t[s:s + 512])[0].cpu().numpy())
    p = np.concatenate(outs)
    _check_pred(p, f"seed={seed}")
    return p, int(sum(q.numel() for q in model.parameters()))


def run(n_seeds: int = 2, epochs: int = 18, diagnostic_002: bool = False, max_folds: int = 0) -> dict:
    import torch  # noqa: F401
    outdir = ENC002_OUT if diagnostic_002 else OUT
    encdir = (outdir / "encoder") if diagnostic_002 else ENC_OUT
    encdir.mkdir(parents=True, exist_ok=True)
    if diagnostic_002:
        pre = ENC002_OUT / "prereg.json"
        pre_obj = json.loads(pre.read_text(encoding="utf-8")) if pre.exists() else {}
        allowed = pre_obj.get("status") == "PRE_REGISTERED_RAW_SEQUENCE_DIAGNOSTIC"
        gate = "PRE_REGISTERED_RAW_SEQUENCE_DIAGNOSTIC" if allowed else "GATE_MISSING"
        force = False
    else:
        dec_path = OUT / "decision.json"
        dec = json.loads(dec_path.read_text(encoding="utf-8")) if dec_path.exists() else {}
        allowed = dec.get("verdict") == "INTRADAY_CANDIDATE_SHADOW_ONLY"
        force = "--force-diagnostic" in sys.argv
        gate = "pass" if allowed else "FORCED_DIAGNOSTIC"
    if not allowed and not force:
        raise SystemExit(f"encoder gated: gate={gate}; use --force-diagnostic for a smoke test")
    t0 = time.time()
    log(f"ENCODER start gate={gate} seeds={n_seeds} epochs={epochs} diagnostic_002={diagnostic_002}")
    X, meta = build_tensors()
    meta.to_parquet(DATA / ("encoder_index_002.parquet" if diagnostic_002 else "encoder_index.parquet"), index=False)
    np.save(DATA / ("encoder_tensors_002.npy" if diagnostic_002 else "encoder_tensors.npy"), X)
    keep = meta["has_tensor"].to_numpy() & meta[TARGET].notna().to_numpy()
    if diagnostic_002:
        # Pair with the regenerated 21-fold LightGBM comparator on identical
        # (date, code) rows; the saved quick-fold B only covered 262 days.
        base = pd.read_parquet(ENC002_BASE / "base_frame_keys.parquet", columns=["date", "code"])
        base["date"] = pd.to_datetime(base["date"]); base["code"] = base["code"].astype(str)
        keys = set(zip(base["date"], base["code"]))
        in_base = np.array([(d, c) in keys for d, c in zip(meta["date"], meta["code"])])
        log(f"  comparator rows={len(base):,} keys={len(keys):,} matched={int((keep & in_base).sum()):,}")
        keep = keep & in_base
    meta = meta[keep].reset_index(drop=True)
    X = X[keep]
    page = pd.read_parquet(DATA / "page_dataset.parquet")
    page["date"] = pd.to_datetime(page["date"]); page["code"] = page["code"].astype(str)
    rename = {c: f"feat__{c}" for c in (DAILY_FEATURES + MARKET_FEATURES) if c in page.columns}
    page = page.rename(columns=rename)
    daily = page.merge(meta[["date", "code"]], on=["date", "code"], how="inner")
    daily = daily.sort_values(["date", "code"]).reset_index(drop=True)
    assert len(daily) == len(meta), (len(daily), len(meta))
    log(f"  tensors ready {X.shape} rows={len(meta):,} dates={meta['date'].nunique()}")
    fl = folds(pd.DatetimeIndex(sorted(daily["date"].unique())), False)
    n_params = None
    preds = []
    done_folds = 0
    for (te_start, te_end) in fl:
        if max_folds and done_folds >= max_folds:
            log(f"  max_folds={max_folds} reached; stopping early")
            break
        all_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
        j = int(all_dates.get_loc(te_start))
        cutoff = all_dates[max(0, j - PURGE_DAYS)]
        tr = daily["date"] <= cutoff
        te = (daily["date"] >= te_start) & (daily["date"] <= te_end)
        if tr.sum() < 500 or te.sum() == 0:
            continue
        fold_path = encdir / f"fold_{te_start.date()}.parquet" if diagnostic_002 else None
        if fold_path is not None and fold_path.exists():
            if _ckpt_ok(fold_path):
                out = pd.read_parquet(fold_path)
                preds.append(out)
                done_folds += 1
                log(f"  fold {te_start.date()} resumed from checkpoint")
                continue
            log(f"  fold {te_start.date()} invalid checkpoint (non-finite/constant pred) -> retrain")
            fold_path.unlink()
        Xtr = X[tr.to_numpy()].astype(np.float32)
        Xte = X[te.to_numpy()].astype(np.float32)
        Dtr, Dte = _prep_daily(select_X(daily[tr], "daily").to_numpy(dtype=np.float64),
                               select_X(daily[te], "daily").to_numpy(dtype=np.float64))
        ytr = daily.loc[tr, TARGET].to_numpy(dtype=np.float32)
        P = np.zeros((n_seeds, int(te.sum())), dtype=np.float32)
        for s in range(n_seeds):
            pi, n_params = train_fold(Xtr, Dtr, ytr, Xte, Dte, seed=s, epochs=epochs)
            P[s] = pi
            log(f"  fold {te_start.date()} seed {s+1}/{n_seeds} done")
        p = P.mean(axis=0)
        _check_pred(p, f"fold={te_start.date()} avg")
        out = daily.loc[te, ["date", "code", "rank_in_fp", TARGET, "label__exec_T1_T2"]].copy()
        out["pred"] = p; out["arm"] = "C_encoder"; out["fold"] = str(te_start.date())
        for s in range(n_seeds):
            out[f"pred_s{s}"] = P[s]
        preds.append(out)
        done_folds += 1
        if fold_path is not None:
            out.to_parquet(fold_path, index=False)
        try:
            fold_ic = float(daily_metrics(out)["ic"].mean())
        except Exception:
            fold_ic = float("nan")
        log(f"  fold {te_start.date()} train={int(tr.sum()):,} test={int(te.sum()):,} "
            f"params={n_params:,} ic={fold_ic:.4f}")
    if n_params is None and preds:
        n_params = 63146
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(DATA / ("oos_pred_C_002.parquet" if diagnostic_002 else "oos_pred_C.parquet"), index=False)
    dc = daily_metrics(pred)
    dc.to_csv(encdir / "daily_metrics_C.csv", index=False, encoding="utf-8-sig")
    b_path = (ENC002_BASE / "oos_pred_B_21f.parquet") if diagnostic_002 else (DATA / "oos_pred_B.parquet")
    cmp = {}
    if b_path.exists():
        db = daily_metrics(pd.read_parquet(b_path))
        m = dc.merge(db, on="date", suffixes=("_C", "_B"))
        if len(m) >= 20:
            d = (m["ic_C"] - m["ic_B"]).to_numpy()
            bt = boot_p(d)
            cmp = {"n_days": int(len(m)), "ic_C": float(dc["ic"].mean()), "ic_B": float(db["ic"].mean()),
                   "delta_ic_C_minus_B": bt["mean"], "p": bt["p_one_sided"], "ci95": bt["ci95"],
                   "top1_C": float(dc["top1_alpha"].mean()), "top1_B": float(db["top1_alpha"].mean())}
        else:
            cmp = {"n_days": int(len(m)), "status": "INSUFFICIENT_OVERLAP_FOR_COMPARISON"}
    res = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "status": "ENCODER_DIAGNOSTIC_ONLY" if diagnostic_002 else ("ENCODER_DIAGNOSTIC_ONLY" if allowed else "FORCED_DIAGNOSTIC"),
           "experiment_id": "MICRO-005-ENC-002" if diagnostic_002 else "MICRO-005-ENCODER",
           "gate": gate,
           "params": n_params, "max_params": MAX_PARAMS,
           "rows": int(len(meta)), "dates": int(meta["date"].nunique()),
           "ic_C": float(dc["ic"].mean()), "top1_C": float(dc["top1_alpha"].mean()),
           "comparison_vs_B": cmp, "elapsed_min": round((time.time() - t0) / 60, 1),
           "production_write": False,
           "note": ("Historical diagnostic only. Even a positive result cannot promote the model; "
                    "the encoder needs >=250 true forward days and 25bps stress first.")}
    (encdir / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                                          encoding="utf-8")
    log("ENCODER_DONE " + json.dumps({k: res[k] for k in ("params", "ic_C", "top1_C")}, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--force-diagnostic", action="store_true")
    ap.add_argument("--diagnostic-002", action="store_true")
    ap.add_argument("--max-folds", type=int, default=0)
    a = ap.parse_args()
    run(a.seeds, a.epochs, a.diagnostic_002, a.max_folds)
