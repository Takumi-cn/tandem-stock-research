"""申万二级行业实时排行与成分股数据源（轻量 requests）。

每次打开界面只拉取一次行业行情；成分股在悬停时按需拉取，不周期刷新。
界面先读内存/磁盘缓存，网络请求在一次性线程中完成，不阻塞 Qt 主线程。
数据：申万宏源研究二级行业实时指数 + 申万成分 + 腾讯个股快照。
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
import urllib3
from pathlib import Path

import requests
from PySide6.QtCore import QObject, Qt, Signal

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "reports" / "industry_board" / "cache.json"
RADAR = ROOT / "reports" / "industry_radar" / "latest.json"
UA = {"User-Agent": "Mozilla/5.0"}
SW_URL = "https://www.swsresearch.com/institute-sw/api/index_publish/current/"
CONS_URL = "https://www.swsresearch.com/institute-sw/api/index_publish/details/component_stocks/"
TENCENT_URL = "https://qt.gtimg.cn/q="
urllib3.disable_warnings()
QUOTE_TTL = 60.0


def _get_json(url: str, params: dict, timeout: float = 12.0) -> dict:
    last: Exception | None = None
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, headers=UA, verify=False, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt == 0:
                time.sleep(0.8)
    raise last or RuntimeError("request failed")


def _num(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _sw_industries() -> list[dict]:
    first = _get_json(SW_URL, {"page": 1, "page_size": 50, "indextype": "二级行业"})
    data = first.get("data") or {}
    rows = list(data.get("results") or [])
    try:
        count = int(data.get("count") or len(rows))
    except (TypeError, ValueError):
        count = len(rows)
    for page in range(2, max(1, math.ceil(count / 50)) + 1):
        payload = _get_json(SW_URL, {"page": page, "page_size": 50, "indextype": "二级行业"})
        rows.extend((payload.get("data") or {}).get("results") or [])
    out: list[dict] = []
    for row in rows:
        code = str(row.get("swindexcode") or "").strip()
        name = str(row.get("swindexname") or "").strip()
        if not code or not name:
            continue
        prev = _num(row.get("l3"))
        last = _num(row.get("l8"))
        if not prev or last is None or prev <= 0:
            continue
        out.append({
            "code": code,
            "name": name,
            "prev": round(prev, 3),
            "last": round(last, 3),
            "pct": round(last / prev - 1.0, 6),
            "amount": round(_num(row.get("l5"), 0.0) or 0.0, 2),
        })
    out.sort(key=lambda x: x["pct"], reverse=True)
    return out


def fetch_snapshot() -> dict:
    industries = _sw_industries()
    now = time.localtime()
    closed = now.tm_wday >= 5 or now.tm_hour * 60 + now.tm_min >= 15 * 60
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trade_state": "收盘" if closed else "盘中",
        "source": "申万宏源研究实时指数 + 申万成分 + 腾讯个股快照",
        "industries": industries,
    }


def _tencent_symbol(code: str) -> str:
    if code.startswith(("6", "9", "5")):
        return "sh" + code
    if code.startswith(("0", "3")):
        return "sz" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sh" + code


def _tencent_quotes(codes: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    codes = [c for c in codes if len(c) == 6]
    for start in range(0, len(codes), 60):
        chunk = codes[start:start + 60]
        query = ",".join(_tencent_symbol(c) for c in chunk)
        r = requests.get(
            TENCENT_URL + query,
            headers={**UA, "Referer": "https://gu.qq.com/"},
            timeout=10.0,
        )
        r.raise_for_status()
        text = r.content.decode("gbk", "replace")
        for line in text.splitlines():
            if '="' not in line:
                continue
            body = line.split('="', 1)[1].rsplit('";', 1)[0]
            parts = body.split("~")
            if len(parts) < 5:
                continue
            code = str(parts[2]).strip()
            if len(code) != 6:
                continue
            current = _num(parts[3])
            prev = _num(parts[4])
            pct = (current / prev - 1.0) if (current is not None and prev and prev > 0) else None
            out[code] = {"pct": pct, "price": current, "name": parts[1]}
    return out


def _components(code: str) -> list[dict]:
    payload = _get_json(CONS_URL, {"swindexcode": code, "page": 1, "page_size": 10000})
    rows = ((payload.get("data") or {}).get("results") or [])
    stocks: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        c = str(row.get("stockcode") or "").strip()
        if len(c) != 6 or c in seen:
            continue
        seen.add(c)
        stocks.append({
            "code": c,
            "name": str(row.get("stockname") or "").strip(),
            "weight": _num(row.get("newweight")),
        })
    return stocks


def fetch_members(code: str) -> dict:
    stocks = _components(code)
    try:
        quotes = _tencent_quotes([s["code"] for s in stocks])
    except Exception:  # noqa: BLE001
        quotes = {}
    for s in stocks:
        q = quotes.get(s["code"]) or {}
        s["pct"] = q.get("pct")
        s["price"] = q.get("price")
        if q.get("name") and not s.get("name"):
            s["name"] = q.get("name")
    return {
        "code": code,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quotes_ts": time.time(),
        "stocks": stocks,
    }


def fetch_quotes_for(code: str, stocks: list[dict]) -> dict:
    quotes = _tencent_quotes([str(s.get("code") or "") for s in stocks])
    return {"code": code, "quotes_ts": time.time(), "quotes": quotes}


def _fallback_cache() -> dict:
    if not RADAR.exists():
        return {}
    try:
        d = json.loads(RADAR.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    industries = []
    for row in d.get("top") or []:
        try:
            industries.append({
                "code": str(row.get("code") or ""),
                "name": str(row.get("name") or ""),
                "pct": float(row.get("r1") or 0.0),
                "prev": None,
                "last": None,
                "amount": float(row.get("amount") or 0.0),
            })
        except (TypeError, ValueError):
            continue
    industries.sort(key=lambda x: x["pct"], reverse=True)
    return {
        "generated_at": str(d.get("as_of") or "") + " 收盘",
        "trade_state": "收盘",
        "source": "industry_radar 缓存（离线降压）",
        "industries": industries,
        "members": {},
    }


class IndustryFeed(QObject):
    snapshot_ready = Signal(dict)
    members_ready = Signal(str)
    error = Signal(str)
    _snapshot_result = Signal(object)
    _members_result = Signal(object)
    _quotes_result = Signal(object)
    _snapshot_err = Signal(str)
    _member_err = Signal(str, str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.cache_path = CACHE
        self.cache = self._load()
        self._members: dict[str, dict] = {
            str(k): v for k, v in (self.cache.get("members") or {}).items()
        }
        self._pending: set[str] = set()
        self._refreshing = False
        self._closing = False
        self._snapshot_result.connect(self._snapshot_done, Qt.ConnectionType.QueuedConnection)
        self._members_result.connect(self._members_done, Qt.ConnectionType.QueuedConnection)
        self._quotes_result.connect(self._quotes_done, Qt.ConnectionType.QueuedConnection)
        self._snapshot_err.connect(self._snapshot_error, Qt.ConnectionType.QueuedConnection)
        self._member_err.connect(self._members_error, Qt.ConnectionType.QueuedConnection)
        if "--screenshot" not in sys.argv:
            self.refresh()

    def _load(self) -> dict:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if len(data.get("industries") or []) >= 20:
                    return data
            except Exception:  # noqa: BLE001
                pass
        return _fallback_cache()

    def _save(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_path)
        except OSError:
            pass

    def shutdown(self) -> None:
        self._closing = True

    def _spawn(self, fn, ok_signal, err_signal, args: tuple = (), err_code: str | None = None) -> None:
        def run():
            if self._closing:
                return
            try:
                data = fn(*args)
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e)[:180]}"
                if err_code is not None:
                    err_signal.emit(err_code, msg)
                else:
                    err_signal.emit(msg)
            else:
                if not self._closing:
                    ok_signal.emit(data)
        threading.Thread(target=run, daemon=True, name="industry-feed").start()

    def refresh(self) -> None:
        if self._refreshing or self._closing:
            return
        self._refreshing = True
        self._spawn(fetch_snapshot, self._snapshot_result, self._snapshot_err)

    def _snapshot_done(self, data: dict) -> None:
        self._refreshing = False
        self.cache = {**data, "members": self._members}
        if data.get("source") != "test":
            self._save()
        self.snapshot_ready.emit(self.cache)

    def _snapshot_error(self, message: str) -> None:
        self._refreshing = False
        self.error.emit(message)

    def ensure_members(self, code: str) -> None:
        code = str(code or "")
        if not code or code in self._pending:
            return
        entry = self._members.get(code)
        if entry:
            age = time.time() - float(entry.get("quotes_ts") or 0.0)
            if age < QUOTE_TTL:
                return
            self._pending.add(code)
            self._spawn(fetch_quotes_for, self._quotes_result, self._member_err, (code, entry.get("stocks") or []), code)
        else:
            self._pending.add(code)
            self._spawn(fetch_members, self._members_result, self._member_err, (code,), code)

    def _members_done(self, data: dict) -> None:
        code = str(data.get("code") or "")
        self._pending.discard(code)
        self._members[code] = {
            "fetched_at": data.get("fetched_at"),
            "quotes_ts": data.get("quotes_ts") or time.time(),
            "stocks": data.get("stocks") or [],
        }
        self.cache["members"] = self._members
        self._save()
        self.members_ready.emit(code)

    def _quotes_done(self, data: dict) -> None:
        code = str(data.get("code") or "")
        self._pending.discard(code)
        entry = self._members.get(code)
        if entry is None:
            return
        quotes = data.get("quotes") or {}
        for s in entry.get("stocks") or []:
            q = quotes.get(str(s.get("code") or "")) or {}
            if q:
                s["pct"] = q.get("pct")
                s["price"] = q.get("price")
        entry["quotes_ts"] = data.get("quotes_ts") or time.time()
        self._members[code] = entry
        self.cache["members"] = self._members
        self._save()
        self.members_ready.emit(code)

    def _members_error(self, code: str, message: str) -> None:
        self._pending.discard(code)
        self.error.emit(f"{code}: {message}")

    def stocks_for(self, code: str) -> list[dict]:
        raw = (self._members.get(str(code)) or {}).get("stocks") or []
        out = [{
            "code": str(s.get("code") or ""),
            "name": str(s.get("name") or ""),
            "pct": s.get("pct"),
            "price": s.get("price"),
        } for s in raw]
        out.sort(key=lambda x: (x["pct"] is None, -(x["pct"] or 0.0)))
        return out

    def has_members(self, code: str) -> bool:
        return str(code) in self._members

    def is_loading(self, code: str) -> bool:
        return str(code) in self._pending


def main() -> None:
    old = {}
    if CACHE.exists():
        try:
            old = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = {}
    try:
        data = fetch_snapshot()
    except Exception as e:  # noqa: BLE001
        if len(old.get("industries") or []) < 20:
            fallback = _fallback_cache()
            if not fallback.get("industries"):
                raise
            old = fallback
        data = {k: v for k, v in old.items() if k != "members"}
        print(f"[industry_feed] 指数接口暂不可用，沿用缓存：{type(e).__name__}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    members = dict(old.get("members") or {})
    for item in (data.get("industries") or [])[:12]:
        code = item["code"]
        cached = members.get(code)
        if cached:
            try:
                quotes = _tencent_quotes([s.get("code") for s in cached.get("stocks") or []])
                for s in cached.get("stocks") or []:
                    q = quotes.get(str(s.get("code") or ""))
                    if q:
                        s["pct"], s["price"] = q.get("pct"), q.get("price")
                cached["quotes_ts"] = time.time()
                cached["fetched_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                members[code] = cached
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                members[code] = fetch_members(code)
            except Exception:  # noqa: BLE001
                continue
    data["members"] = members
    CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print("INDUSTRY_FEED_DONE", len(data.get("industries") or []), len(members), data.get("generated_at"))


if __name__ == "__main__":
    main()
