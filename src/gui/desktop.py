"""每日双轨预测 · 原生桌面界面（PySide6，无浏览器依赖）。

启动：pythonw run_gui.pyw
交互：开始预测（子进程跑完整管线）｜丝滑悬浮进度｜结果自动刷新
测试：python -m src.gui.desktop --screenshot out.png [--size 1400x1600]
"""
from __future__ import annotations

import csv
import ctypes
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import (
    QEvent, QEasingCurve, QIODevice, QObject, QPoint, QPointF, Property, QPropertyAnimation,
    QProcess, QProcessEnvironment, QRect, QRectF, Qt, QTimer, QUrl, QVariantAnimation, Signal,
)
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QDesktopServices, QFont, QIcon, QKeySequence, QLinearGradient, QPainter, QPainterPath,
    QPen, QPixmap, QRadialGradient, QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QFrame, QGraphicsOpacityEffect,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizeGrip, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from src.gui.industry_feed import IndustryFeed

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
LATEST = ROOT / "reports" / "latest.json"
LATEST_GUESS = ROOT / "reports" / "latest_guess.json"
PROG = ROOT / "reports" / "runs" / "progress.jsonl"
RUN_LOG = ROOT / "reports" / "runs" / "gui_run.log"
NAMES_FILE = ROOT / "data" / "raw" / "stock_names.json"
try:
    NAMES = json.loads(NAMES_FILE.read_text(encoding="utf-8")) if NAMES_FILE.exists() else {}
except Exception:  # noqa: BLE001
    NAMES = {}

C = {
    "bg": "#F7F4FC", "card": "#FFFFFF", "ink": "#16121F", "ink2": "#4E4759",
    "muted": "#8D869A", "line": "#ECE7F4", "lav": "#F4EFFF", "lav2": "#E8DEFF",
    "purple": "#7C3AED", "purple_d": "#5B21B6", "purple_l": "#A78BFA",
    "warn": "#B45309", "warn_bg": "#FFF7E8", "warn_line": "#F6E2BD",
    "ok": "#6D28D9", "down": "#5E5868", "down_bg": "#F2F0F5",
}

FONT_CN = "Microsoft YaHei UI"
FONT_NUM = "Segoe UI"


def _font(size: float, weight: QFont.Weight = QFont.Weight.Normal, cn: bool = False, spacing: float | None = None) -> QFont:
    f = QFont(FONT_CN if cn else FONT_NUM)
    f.setPixelSize(max(1, int(round(size))))
    f.setWeight(weight)
    if spacing is not None:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


def _label(text: str = "", font: QFont | None = None, color: str = C["ink"], wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setTextFormat(Qt.TextFormat.PlainText)
    if font:
        lab.setFont(font)
    lab.setStyleSheet(f"color:{color};background:transparent;")
    lab.setWordWrap(wrap)
    return lab


def _pill(text: str, fg: str, bg: str, border: str | None = None, size: int = 12, padding: str = "5px 11px") -> QLabel:
    return IslandPill(text, fg, bg, border, size, padding)


def _display_industry(name: str) -> str:
    """申万二级的 Ⅱ/Ⅲ 是层级标记，界面上去掉，避免被读成数字 2/3。"""
    text = str(name or "").strip()
    for suffix in ("Ⅱ", "Ⅲ", "II", "III"):
        if text.endswith(suffix):
            return text[: -len(suffix)].rstrip()
    return text


def _fmt_pct(v: float | None, digits: int = 1, signed: bool = False) -> str:
    if v is None or not math.isfinite(float(v)):
        return "—"
    sign = "+" if signed and v > 0 else ""
    return f"{sign}{float(v) * 100:.{digits}f}%"


def _fmt_pp(v: float | None, digits: int = 2, signed: bool = True) -> str:
    """已是百分点（如 10.01 表示 +10.01%）。"""
    if v is None:
        return "—"
    sign = "+" if signed and v > 0 else ""
    return f"{sign}{float(v):.{digits}f}%"


def _auto_mode() -> bool:
    """全自动运行时只写日志，不弹窗阻塞。"""
    return os.environ.get("TANDEM_AUTO") == "1" or "--auto" in sys.argv


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _read_progress() -> dict:
    out = {"pct": 0.0, "stage": "", "msg": ""}
    if not PROG.exists():
        return out
    try:
        lines = PROG.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            d = json.loads(lines[-1])
            out["pct"] = float(d.get("pct", 0))
            out["stage"] = d.get("stage", "")
            out["msg"] = d.get("msg", "")
    except Exception:  # noqa: BLE001
        pass
    return out


def _clear_layout(layout) -> None:
    """Recursively remove widgets AND nested layouts (the old code leaked layouts)."""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
            continue
        child = item.layout()
        if child is not None:
            _clear_layout(child)
            child.deleteLater()


def _fade_in(widget: QWidget, delay: int = 0, duration: int = 240) -> None:
    """One-shot opacity reveal; starts at 0.45 to avoid a blank flash on refresh."""
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.45)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    widget._reveal_anim = anim
    widget._reveal_eff = eff

    def _done():
        try:
            widget.setGraphicsEffect(None)
        except RuntimeError:
            pass

    anim.finished.connect(_done)
    QTimer.singleShot(delay, anim.start)


def _spring(t: float, response: float = 0.42, damping: float = 0.86) -> float:
    """Damped-spring easing (0..1 with a small, decaying overshoot).

    Faster attack and more damping than the old 0.50/0.74 spring: it still
    feels alive, but settles once instead of visibly wobbling.
    `t` is normalised time.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    w0 = 2.0 * math.pi / max(0.05, response)
    wd = w0 * math.sqrt(max(1e-9, 1.0 - damping * damping))
    env = math.exp(-damping * w0 * t)
    return 1.0 - env * (math.cos(wd * t) + (damping * w0 / wd) * math.sin(wd * t))


class _FrameBus(QObject):
    """One 60 Hz clock for every ambient effect.

    All widgets receive the same frame and mark themselves dirty inside one
    event-loop tick, so Qt coalesces them into a single paint pass instead of
    hundreds of independent timer wake-ups per second.
    """

    tick = Signal(int)

    def __init__(self):
        super().__init__()
        self._frame = 0
        self._timer = QTimer(self)
        # 45 fps for the light surfaces; large panels update on every other
        # tick. Faster than this hits the 60 Hz compositor cliff (85–94% CPU).
        self._timer.setInterval(22)
        self._timer.timeout.connect(self._emit)

    def subscribe(self, slot) -> None:
        self.tick.connect(slot)
        if not self._timer.isActive():
            self._timer.start()

    def pause(self) -> None:
        self._timer.stop()

    def resume(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _emit(self) -> None:
        self._frame += 1
        self.tick.emit(self._frame)


_FRAME = _FrameBus()


class IslandCapsule(QWidget):
    """Dynamic-Island style capsule that grows out of a badge over the title block.

    Transparent to the mouse, painted on top of the card's title labels, clipped
    to its current width so the text is revealed exactly as the capsule sweeps
    over it.  Never changes layout.
    """

    def __init__(self, badge: "HoverBadge", title: QLabel, sub: QLabel | None, card: QWidget):
        super().__init__(card)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._badge, self._title, self._sub, self._card = badge, title, sub, card
        self._t = 0.0
        self._target = 0.0
        self._span = QRectF()
        self._title_rect = QRectF()
        self._sub_rect = QRectF()
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(430)
        self._anim.valueChanged.connect(self._step)
        self.hide()

    # ---- geometry -------------------------------------------------
    def _relayout(self) -> None:
        card = self._card
        b = QRectF(QRect(self._badge.mapTo(card, QPoint(0, 0)), self._badge.size()))
        tr = QRectF(QRect(self._title.mapTo(card, QPoint(0, 0)), self._title.size()))
        right, bottom = tr.right(), tr.bottom()
        top = min(b.top(), tr.top())
        sr = None
        if self._sub is not None:
            sr = QRectF(QRect(self._sub.mapTo(card, QPoint(0, 0)), self._sub.size()))
            right, bottom = max(right, sr.right()), max(bottom, sr.bottom())
        x0 = b.right() - 2.0
        h = max(b.height(), bottom - top) + 8.0
        self._span = QRectF(x0, top - 4.0, max(12.0, right - x0 + 14.0), h)
        self._title_rect = QRectF(tr.x() - x0, tr.y() - self._span.y(), tr.width(), tr.height())
        self._sub_rect = (QRectF(sr.x() - x0, sr.y() - self._span.y(), sr.width(), sr.height())
                          if sr is not None else QRectF())
        self.setGeometry(int(self._span.x()), int(self._span.y()),
                         int(math.ceil(self._span.width())), int(math.ceil(self._span.height())))
        self.raise_()

    # ---- animation ------------------------------------------------
    def expand(self) -> None:
        self._relayout()
        self.show()
        self._target = 1.0
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def collapse(self) -> None:
        self._target = 0.0
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(0.0)
        self._anim.start()

    def _step(self, value) -> None:
        self._t = float(value)
        self.update()
        if self._target == 0.0 and self._t <= 0.002:
            self.hide()

    # ---- painting -------------------------------------------------
    def paintEvent(self, ev):  # noqa: N802
        if self._t <= 0.0 or self._span.isEmpty():
            return
        pe = max(0.0, min(1.06, _spring(self._t, response=0.62, damping=0.86)))
        h = self._span.height()
        w = max(4.0, self._span.width() * pe)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setClipRect(QRectF(0, 0, w, h))
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(241, 234, 255, 252))
        grad.setColorAt(0.55, QColor(247, 243, 255, 252))
        grad.setColorAt(1.0, QColor(250, 247, 255, 252))
        p.setPen(QPen(QColor(226, 213, 247, 210), 1.0))
        p.setBrush(QBrush(grad))
        radius = h / 2.0
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), radius, radius)
        # title / sub, exactly where the real labels sit
        p.setPen(QColor(C["ink"]))
        p.setFont(self._title.font())
        p.drawText(self._title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self._title.text())
        if self._sub is not None and not self._sub_rect.isEmpty():
            p.setPen(QColor(C["muted"]))
            p.setFont(self._sub.font())
            p.drawText(self._sub_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       self._sub.text())


class HoverBadge(QWidget):
    """Letter badge with a liquid spring: squash/stretch + corner morph."""

    def __init__(self, text: str, width: int = 34, height: int = 34, radius: int = 11,
                 bg: str = "#F0E9FF", fg: str = "#6D28D9", font_size: float = 15,
                 pad: int = 3, scale: float = 0.08, bold: bool = True):
        super().__init__()
        self._text = text
        self._bw, self._bh = width, height
        self._radius = radius
        self._bg = QColor(bg)
        self._fg = QColor(fg)
        self._font = _font(font_size, QFont.Weight.Bold if bold else QFont.Weight.DemiBold)
        self._scale = scale
        self._t = 0.0
        self._target = 0.0
        self._island: IslandCapsule | None = None
        self._island_title: QLabel | None = None
        self._island_sub: QLabel | None = None
        self.setFixedSize(width + pad * 2, height + pad * 2)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(320)
        self._anim.valueChanged.connect(self._step)
        self._phase = (id(self) % 628) / 100.0
        _FRAME.subscribe(self._breathe)

    def _breathe(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.034) % (math.pi * 2)
        self.update()

    # ---- island wiring --------------------------------------------
    def attach_island(self, title: QLabel, sub: QLabel | None) -> None:
        self._island_title, self._island_sub = title, sub
        self._island_open = False
        # keep the capsule open while the pointer is anywhere on badge + capsule
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(33)
        self._hover_timer.timeout.connect(self._check_hover)

    def _ensure_island(self) -> None:
        if self._island is not None or self._island_title is None:
            return
        card = self.parentWidget()
        while card is not None and card.objectName() not in ("Card", "SectionBar"):
            card = card.parentWidget()
        if card is None:
            return
        self._island = IslandCapsule(self, self._island_title, self._island_sub, card)

    # ---- animation ------------------------------------------------
    def _animate(self, target: float) -> None:
        self._target = target
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(target)
        self._anim.start()

    def _step(self, value) -> None:
        self._t = float(value)
        self.update()

    def enterEvent(self, ev):  # noqa: N802
        self._ensure_island()
        self._animate(1.0)
        if self._island is not None:
            self._island_open = True
            self._island.expand()
            self._hover_timer.start()
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802
        if self._island is None:
            self._animate(0.0)
        super().leaveEvent(ev)

    def _check_hover(self) -> None:
        if not self._island_open:
            self._hover_timer.stop()
            return
        pos = QCursor.pos()
        on_badge = self.rect().contains(self.mapFromGlobal(pos))
        on_island = (self._island is not None and self._island.isVisible()
                     and self._island.rect().contains(self._island.mapFromGlobal(pos)))
        if on_badge or on_island:
            return
        self._island_open = False
        self._hover_timer.stop()
        self._animate(0.0)
        if self._island is not None:
            self._island.collapse()

    # ---- painting -------------------------------------------------
    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        e = _spring(self._t)
        cx, cy = self.width() / 2, self.height() / 2
        # slight squash & stretch: x leads, y follows
        sx = 1.0 + self._scale * 1.15 * e
        sy = 1.0 + self._scale * 0.75 * e
        w, h = self._bw * sx, self._bh * sy
        rect = QRectF(cx - w / 2, cy - h / 2, w, h)
        grow = max(0.0, e)
        # corner morph: rounded square -> capsule at peak
        cap = min(w, h) / 2.0
        radius = self._radius + (cap - self._radius) * min(1.0, grow)
        breathe = 0.5 + 0.5 * math.sin(self._phase)
        p.setPen(Qt.PenStyle.NoPen)
        if grow > 0.01:
            p.setBrush(QColor(124, 58, 237, int(22 * min(1.0, grow))))
            p.drawRoundedRect(rect.adjusted(-3.5, -3.5, 3.5, 3.5), radius + 3.5, radius + 3.5)
        else:
            p.setBrush(QColor(124, 58, 237, int(9 + 9 * breathe)))
            p.drawRoundedRect(rect.adjusted(-3, -3, 3, 3), radius + 3, radius + 3)
        p.setBrush(self._bg)
        p.drawRoundedRect(rect, radius, radius)
        # slow diagonal glint inside the badge
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        p.save()
        p.setClipPath(path)
        gx = rect.left() - rect.width() + rect.width() * 2.0 * (0.5 + 0.5 * math.sin(self._phase * 0.72))
        sheen = QLinearGradient(gx - 16, 0, gx + 16, 0)
        sheen.setColorAt(0.0, QColor(255, 255, 255, 0))
        sheen.setColorAt(0.5, QColor(255, 255, 255, 46))
        sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(rect, sheen)
        p.restore()
        # a tiny satellite circles the badge
        ang = self._phase * 1.25
        orbit = min(rect.width(), rect.height()) / 2.0 + 2.0
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(124, 58, 237, int(65 + 35 * breathe)))
        p.drawEllipse(QPointF(cx + orbit * math.cos(ang), cy + orbit * math.sin(ang)), 1.5, 1.5)
        p.setPen(self._fg)
        p.setFont(self._font)
        # letter counter-scales a little so it does not look stretched
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._text)


# ------------------------------------------------------------------ 通用视觉组件
class IslandPill(QLabel):
    """Pill chip that springs into a capsule under the cursor (island language)."""

    def __init__(self, text: str, fg: str, bg: str, border: str | None = None,
                 size: float = 12, padding: str = "5px 11px"):
        super().__init__(text)
        self._fg = QColor(fg)
        self._bg = QColor(bg)
        self._border = QColor(border) if border else None
        self._radius = 9.0
        self._inset = 2
        self.setFont(_font(size, QFont.Weight.DemiBold, cn=True))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        parts = [p for p in padding.replace("px", "").split() if p]
        self._pad_v = int(float(parts[0])) if parts else 5
        self._pad_h = int(float(parts[-1])) if parts else 11
        self._t = 0.0
        self._target = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(300)
        self._anim.valueChanged.connect(self._step)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._phase = (id(self) % 628) / 100.0
        _FRAME.subscribe(self._breathe)
        self._fit()

    def _breathe(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.020) % (math.pi * 2)
        self.update()

    def _fit(self) -> None:
        fm = self.fontMetrics()
        tw = fm.horizontalAdvance(self.text())
        self.setFixedSize(tw + 2 * self._pad_h + 2 * self._inset + 1,
                          fm.height() + 2 * self._pad_v + 2 * self._inset)

    def setText(self, text: str) -> None:  # noqa: D401 - QLabel override
        super().setText(text)
        self._fit()

    def _step(self, value) -> None:
        self._t = float(value)
        self.update()

    def _spring_to(self, target: float) -> None:
        self._target = target
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(target)
        self._anim.start()

    def enterEvent(self, ev):  # noqa: N802
        self._spring_to(1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802
        self._spring_to(0.0)
        super().leaveEvent(ev)

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        e = _spring(self._t)
        base = QRectF(self._inset, self._inset, self.width() - 2 * self._inset,
                      self.height() - 2 * self._inset)
        s = 1.0 + 0.040 * e
        cx, cy = base.center().x(), base.center().y()
        w, h = base.width() * s, base.height() * s
        rect = QRectF(cx - w / 2, cy - h / 2, w, h)
        radius = self._radius + (h / 2.0 - self._radius) * min(1.0, max(0.0, e))
        if e > 0.01:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(124, 58, 237, int(20 * min(1.0, e))))
            p.drawRoundedRect(rect.adjusted(-2, -2, 2, 2), radius + 2, radius + 2)
        p.setBrush(self._bg)
        if self._border is not None:
            p.setPen(QPen(self._border, 1.0))
        else:
            p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, radius, radius)
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        p.save()
        p.setClipPath(path)
        gx = rect.left() - rect.width() + rect.width() * 2.0 * (0.5 + 0.5 * math.sin(self._phase * 0.6))
        sheen = QLinearGradient(gx - 18, 0, gx + 18, 0)
        sheen.setColorAt(0.0, QColor(255, 255, 255, 0))
        if self._bg.lightness() < 150:
            sheen.setColorAt(0.5, QColor(255, 255, 255, 44))
        else:
            sheen.setColorAt(0.5, QColor(124, 58, 237, 22))
        sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(rect, sheen)
        p.restore()
        p.setPen(self._fg)
        p.setFont(self.font())
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())


class IslandButton(QPushButton):
    """Compact button that springs into a capsule on hover / press."""

    _STYLE = {
        "primary": {"radius": 12.0, "text": "#FFFFFF", "size": 12.5, "weight": QFont.Weight.DemiBold,
                    "grad": ("#8B5CF6", "#6D28D9"), "grad_hover": ("#7C3AED", "#5B21B6"),
                    "disabled": ("#D8CCF2", "#D8CCF2"), "border": None, "bg": None,
                    "hover_bg": None, "hover_text": None, "glow": 40},
        "ghost": {"radius": 12.0, "text": C["ink2"], "size": 12.0, "weight": QFont.Weight.Normal,
                  "grad": None, "grad_hover": None, "disabled": None, "border": C["line"], "bg": "#FFFFFF",
                  "hover_bg": "#FAF7FF", "hover_text": None, "glow": 26},
        "copy": {"radius": 8.0, "text": C["purple_d"], "size": 10.5, "weight": QFont.Weight.DemiBold,
                 "grad": None, "grad_hover": None, "disabled": None, "border": "#E7DDF7", "bg": "#F7F2FF",
                 "hover_bg": "#EFE7FF", "hover_text": None, "glow": 26},
        "mini": {"radius": 13.0, "text": C["purple_d"], "size": 10.0, "weight": QFont.Weight.DemiBold,
                 "grad": None, "grad_hover": None, "disabled": None, "border": "#E7DDF7", "bg": "#FFFFFF",
                 "hover_bg": "#F4EEFF", "hover_text": C["purple_d"], "glow": 22},
        # window chrome
        "win": {"radius": 9.0, "text": "#6B6478", "size": 12.0, "weight": QFont.Weight.DemiBold,
                "grad": None, "grad_hover": None, "disabled": None, "border": None, "bg": "#FFFFFF",
                "hover_bg": "#F2EDFA", "hover_text": C["purple_d"], "glow": 0},
        "close": {"radius": 9.0, "text": "#8D869A", "size": 14.0, "weight": QFont.Weight.Normal,
                  "grad": None, "grad_hover": None, "disabled": None, "border": None, "bg": "#FFFFFF",
                  "hover_bg": "#FDECEC", "hover_text": "#C53030", "glow": 0},
    }

    def __init__(self, text: str, variant: str = "ghost", parent: QWidget | None = None):
        super().__init__(text, parent)
        self._variant = variant
        self._st = self._STYLE[variant]
        self._hovered = False
        self._pressed = False
        self._t = 0.0
        self._target = 0.0
        self.setFont(_font(self._st["size"], self._st["weight"], cn=True))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(280)
        self._anim.valueChanged.connect(self._step)
        self._ripple = 0.0
        self._ripple_pos = QPointF(0, 0)
        self._ripple_anim = QVariantAnimation(self)
        self._ripple_anim.setDuration(420)
        self._ripple_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._ripple_anim.valueChanged.connect(self._ripple_step)
        self._phase = 0.0
        if variant == "primary":
            _FRAME.subscribe(self._breathe)

    def _breathe(self, frame: int) -> None:
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.034) % (math.pi * 2)
        self.update()

    def _step(self, value) -> None:
        self._t = float(value)
        self.update()

    def _ripple_step(self, value) -> None:
        self._ripple = float(value)
        self.update()

    def _spring_to(self, target: float) -> None:
        self._target = target
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(target)
        self._anim.start()

    def enterEvent(self, ev):  # noqa: N802
        self._hovered = True
        self._spring_to(1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802
        self._hovered = False
        self._spring_to(0.0)
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):  # noqa: N802
        self._pressed = True
        self._ripple_pos = ev.position()
        self._ripple_anim.stop()
        self._ripple_anim.setStartValue(0.0)
        self._ripple_anim.setEndValue(1.0)
        self._ripple_anim.start()
        self.update()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802
        self._pressed = False
        self.update()
        super().mouseReleaseEvent(ev)

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        st = self._st
        enabled = self.isEnabled()
        e = _spring(self._t)
        inset = 3.0
        base = QRectF(inset, inset, self.width() - 2 * inset, self.height() - 2 * inset)
        s = 1.0 + 0.060 * e
        if self._pressed:
            s *= 0.955
        cx, cy = base.center().x(), base.center().y()
        w, h = base.width() * s, base.height() * s
        rect = QRectF(cx - w / 2, cy - h / 2, w, h)
        radius = st["radius"] + (h / 2.0 - st["radius"]) * min(1.0, max(0.0, e)) * 0.55
        glow = int(st.get("glow", 0) or 0)
        if e > 0.01 and enabled and glow:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(109, 40, 217, int((glow + 18) * min(1.0, e))))
            p.drawRoundedRect(rect.adjusted(-2.5, -2.5, 2.5, 3.5), radius + 2.5, radius + 2.5)
        if st["grad"]:
            c0, c1 = st["grad_hover"] if (self._hovered and enabled) else st["grad"]
            if not enabled:
                c0, c1 = st["disabled"]
            g = QLinearGradient(rect.left(), 0, rect.right(), 0)
            g.setColorAt(0.0, QColor(c0))
            g.setColorAt(1.0, QColor(c1))
            p.setBrush(QBrush(g))
            p.setPen(Qt.PenStyle.NoPen)
        else:
            bg = st["bg"]
            bd = st["border"]
            if not enabled:
                bg, bd = "#F5F2FA", "#EDE8F5"
            elif self._hovered:
                bg = st.get("hover_bg") or bg
                if self._variant in ("ghost", "copy"):
                    bd = C["lav2"]
            p.setBrush(QColor(bg))
            if bd:
                p.setPen(QPen(QColor(bd), 1.0))
            else:
                p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, radius, radius)
        # idle breathing + a periodic sheen for the primary action
        if st["grad"] and enabled and not self._hovered:
            breathe = 0.5 + 0.5 * math.sin(self._phase)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(109, 40, 217, int(16 + 14 * breathe)))
            p.drawRoundedRect(rect.adjusted(-2, -2, 2, 3), radius + 2, radius + 2)
            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)
            p.save()
            p.setClipPath(path)
            k = (self._phase % 4.8) / 4.8
            if k < 0.55:
                sx = rect.left() - rect.width() + rect.width() * 2.1 * (k / 0.55)
                sheen = QLinearGradient(sx - 36, 0, sx + 36, 0)
                sheen.setColorAt(0.0, QColor(255, 255, 255, 0))
                sheen.setColorAt(0.5, QColor(255, 255, 255, 54))
                sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
                p.fillRect(rect, sheen)
            p.restore()
        # hover sheen on filled buttons (one sweep per hover, no timer)
        if st["grad"] and self._hovered and enabled:
            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)
            p.save()
            p.setClipPath(path)
            sx = rect.left() - rect.width() + (rect.width() * 2.2) * min(1.0, self._t)
            sheen = QLinearGradient(sx - 34, 0, sx + 34, 0)
            sheen.setColorAt(0.0, QColor(255, 255, 255, 0))
            sheen.setColorAt(0.5, QColor(255, 255, 255, 82))
            sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
            p.fillRect(rect, sheen)
            p.restore()
        # click ripple
        if 0.0 < self._ripple < 1.0:
            path = QPainterPath()
            path.addRoundedRect(rect, radius, radius)
            p.save()
            p.setClipPath(path)
            rr = max(rect.width(), rect.height()) * (0.12 + 0.78 * self._ripple)
            alpha = int((84 if st["grad"] else 48) * (1.0 - self._ripple))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, alpha) if st["grad"] else QColor(124, 58, 237, alpha))
            p.drawEllipse(self._ripple_pos, rr, rr)
            p.restore()
        text_color = st.get("hover_text") if (self._hovered and st.get("hover_text")) else st["text"]
        p.setPen(QColor(text_color if enabled else "#9A93A5"))
        p.setFont(self.font())
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())


class CountUpLabel(QLabel):
    """Number that springs from 0 to its value (percentage by default)."""

    def __init__(self, value, digits: int = 1, font: QFont | None = None,
                 color: str = "#5B21B6", suffix: str = "%", mult: float = 100.0,
                 duration: int = 780):
        self._value = None if value is None else float(value)
        self._digits, self._suffix, self._mult = digits, suffix, mult
        super().__init__("—" if self._value is None else f"{self._value * mult:.{digits}f}{suffix}")
        if font is not None:
            self.setFont(font)
        self.setStyleSheet(f"color:{color};background:transparent;")
        if self._value is None:
            return
        self.setMinimumWidth(self.fontMetrics().horizontalAdvance(f"{100.0 * mult:.{digits}f}{suffix}"))
        self._t = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._step)
        QTimer.singleShot(60, self._start)

    def _start(self):
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _step(self, value):
        self._t = float(value)
        v = (self._value or 0.0) * self._t
        self.setText(f"{v * self._mult:.{self._digits}f}{self._suffix}")


class PulseDot(QWidget):
    """Small purple dot with a breathing halo (Live indicator)."""

    def __init__(self, size: int = 9, color: str = "#7C3AED"):
        super().__init__()
        self._size = size
        self._color = QColor(color)
        self._phase = 0.0
        self.setFixedSize(size + 12, size + 12)
        _FRAME.subscribe(self._beat)

    def _beat(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.073) % (2 * math.pi)
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        breathe = 0.5 + 0.5 * math.sin(self._phase)
        halo = self._size / 2.0 + 1.5 + 2.2 * breathe
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self._color.red(), self._color.green(), self._color.blue(), int(46 + 42 * breathe)))
        p.drawEllipse(QPointF(cx, cy), halo, halo)
        p.setBrush(self._color)
        p.drawEllipse(QPointF(cx, cy), self._size / 2.0, self._size / 2.0)


class ShimmerTitle(QWidget):
    """Headline with a slow band of purple light travelling through the glyphs."""

    def __init__(self, text: str, font: QFont):
        super().__init__()
        self._text = text
        self.setFont(font)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        fm = self.fontMetrics()
        self.setFixedSize(fm.horizontalAdvance(text) + 18, fm.height() + 8)
        self._phase = 0.0
        _FRAME.subscribe(self._beat)

    def _beat(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.040) % (math.pi * 2)
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = self.fontMetrics()
        path = QPainterPath()
        path.addText(2.0, fm.ascent() + 4.0, self.font(), self._text)
        p.setPen(QPen(QColor(124, 58, 237, 28), 5.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        pos = min(0.97, max(0.03, 0.5 + 0.5 * math.sin(self._phase * 0.72)))
        p0 = max(0.002, pos - 0.22)
        p2 = min(0.998, pos + 0.22)
        g = QLinearGradient(0, 0, max(1.0, float(self.width())), 0)
        g.setColorAt(0.0, QColor("#16121F"))
        g.setColorAt(p0, QColor("#3B0764"))
        g.setColorAt(pos, QColor("#7C3AED"))
        g.setColorAt(p2, QColor("#3B0764"))
        g.setColorAt(1.0, QColor("#16121F"))
        p.setPen(Qt.PenStyle.NoPen)
        p.fillPath(path, QBrush(g))
        p.end()


class SectionLine(QWidget):
    """Thin divider with a light pulse travelling along it."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(3)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._phase = 0.0
        _FRAME.subscribe(self._beat)

    def _beat(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.044) % (math.pi * 2)
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, y = float(self.width()), self.height() / 2.0
        p.setPen(QPen(QColor("#EDE8F4"), 1.2))
        p.drawLine(QPointF(0, y), QPointF(w, y))
        cx = w * (0.5 + 0.5 * math.sin(self._phase * 0.9))
        g = QLinearGradient(cx - 90, 0, cx + 90, 0)
        g.setColorAt(0.0, QColor(167, 139, 250, 0))
        g.setColorAt(0.5, QColor(124, 58, 237, 165))
        g.setColorAt(1.0, QColor(167, 139, 250, 0))
        p.setPen(QPen(QBrush(g), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(0, y), QPointF(w, y))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(124, 58, 237, 140))
        p.drawEllipse(QPointF(cx, y), 1.7, 1.7)
        p.end()


class AccentBar(QWidget):
    """Card top accent with a white highlight running across the gradient."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(4)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._phase = 0.0
        _FRAME.subscribe(self._beat)

    def _beat(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.036) % 1.0
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        rect = QRectF(0, 0, w, h)
        g = QLinearGradient(0, 0, w, 0)
        g.setColorAt(0.0, QColor("#C4B5FD"))
        g.setColorAt(0.55, QColor("#8B5CF6"))
        g.setColorAt(1.0, QColor("#6D28D9"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(g))
        p.drawRoundedRect(rect, 2, 2)
        path = QPainterPath()
        path.addRoundedRect(rect, 2, 2)
        p.setClipPath(path)
        cx = -70.0 + (w + 140.0) * self._phase
        sheen = QLinearGradient(cx - 45, 0, cx + 45, 0)
        sheen.setColorAt(0.0, QColor(255, 255, 255, 0))
        sheen.setColorAt(0.5, QColor(255, 255, 255, 150))
        sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(rect, sheen)
        p.end()


class HeroIsland(QWidget):
    """White / lavender expanding Dynamic Island that holds the live decision.

    Collapsed: live dot + T + pick + probability + action.
    Hover / click: the capsule grows downward and reveals runtime, bucket,
    page size, weights and the research disclaimer.
    """

    H0, H1 = 60, 130

    def __init__(self, state: dict, parent: QWidget | None = None, host: QWidget | None = None):
        super().__init__(parent)
        self._t = 0.0
        self._pinned = False
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.H0)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(460)
        self._anim.valueChanged.connect(self._step)
        self._phase = 0.0
        _FRAME.subscribe(self._tick_ambient)
        self._hover_inside = False
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(33)
        self._hover_timer.timeout.connect(self._check_hover)

        short = state.get("short", {}) or {}
        long_ = state.get("long", {}) or {}
        code = str(short.get("code", "") or "")
        name = str(short.get("name", "") or NAMES.get(code, ""))
        p_up = short.get("win_prob", short.get("p_up"))
        action = str(short.get("action", "") or "未运行")
        T = str(state.get("T", "") or "—")
        bucket = str(state.get("bucket", "") or "—")
        runtime = state.get("runtime_s")
        weights = state.get("weights", {}) or {}
        first = short.get("first_page", []) or []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 0, 18, 0)
        lay.setSpacing(0)
        top_host = QWidget()
        top_host.setFixedHeight(self.H0)
        top = QHBoxLayout(top_host)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(12)

        top.addWidget(PulseDot(9))
        sep = QFrame()
        sep.setFixedSize(1, 20)
        sep.setStyleSheet("background:#E7DDF7;border:none;")
        top.addWidget(sep)
        top.addWidget(_label(f"T {T}", _font(11, QFont.Weight.DemiBold, cn=True), "#6B6478"))
        top.addSpacing(4)
        top.addWidget(_label(f"{code} {name}".strip() or "等待首次预测",
                             _font(14, QFont.Weight.Bold, cn=True), "#16121F"))
        top.addWidget(_label("上涨概率", _font(9.5, cn=True), "#8D869A"))
        top.addWidget(CountUpLabel(p_up, 1, _font(23, QFont.Weight.Bold), "#5B21B6"))
        if "不出手" in action:
            fg, bg = "#8D869A", "#F4F1F8"
        elif "试多" in action or "参与" in action:
            fg, bg = "#FFFFFF", "#6D28D9"
        else:
            fg, bg = "#5B21B6", "#EFE8FD"
        top.addWidget(_pill(action, fg, bg, None, 10.5, "5px 12px"))
        top.addStretch(1)
        lay.addWidget(top_host)

        self._detail = QWidget(self)
        self._detail.setStyleSheet("background:transparent;")
        self._detail.setFixedHeight(self.H1 - self.H0)
        self._detail.hide()
        self._detail_eff = QGraphicsOpacityEffect(self._detail)
        self._detail_eff.setOpacity(0.60)
        self._detail.setGraphicsEffect(self._detail_eff)
        dv = QHBoxLayout(self._detail)
        dv.setContentsMargins(20, 10, 20, 10)
        dv.setSpacing(9)

        def chip(text: str) -> QLabel:
            return _pill(text, "#7A7290", "#F7F3FF", "#EEE7F9", 9.6, "4px 9px")

        top_w = sorted(weights.items(), key=lambda kv: -float(kv[1]))[:3]
        dv.addWidget(chip(f"运行 {runtime}s" if runtime else "运行 —"))
        dv.addWidget(chip(f"桶 {bucket}"))
        page_n = short.get("page_size") or (20 if first else "—")
        dv.addWidget(chip(f"第一页 {page_n} 只｜池 {short.get('pool', '—')}"))
        dv.addWidget(chip("权重 " + " / ".join(f"{k} {float(v):.0%}" for k, v in top_w) if top_w else "权重 —"))
        dv.addWidget(chip(f"长线 {long_.get('state', '—')} · {long_.get('aux', '—')}"))
        dv.addStretch(1)
        dv.addWidget(_pill("研究记录 · 不构成投资建议", "#9A93A5", "#FFFFFF", "#EEE7F9", 9.0, "4px 8px"))

    def _tick_ambient(self, frame: int) -> None:
        self._phase = (self._phase + 0.036) % (math.pi * 2)
        if self.isVisible():
            self.update()

    def _check_hover(self) -> None:
        if self._pinned:
            return
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        if inside != self._hover_inside:
            self._hover_inside = inside
            self._to(1.0 if inside else 0.0)

    def _step(self, value) -> None:
        self._t = float(value)
        e = _spring(self._t, response=0.55, damping=0.86)
        self._detail_eff.setOpacity(0.60 + 0.40 * min(1.0, max(0.0, e)))
        self.update()

    def _to(self, target: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(target)
        self._anim.start()

    def enterEvent(self, ev):  # noqa: N802
        self._hover_inside = True
        self._to(1.0)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802
        if not self._pinned:
            self._hover_inside = False
            self._to(0.0)
        super().leaveEvent(ev)

    def mousePressEvent(self, ev):  # noqa: N802
        if ev.button() == Qt.MouseButton.LeftButton:
            self._pinned = not self._pinned
            self._to(1.0 if self._pinned else 0.0)
        super().mousePressEvent(ev)

    def hideEvent(self, ev):  # noqa: N802
        self._hover_timer.stop()
        super().hideEvent(ev)

    def showEvent(self, ev):  # noqa: N802
        self._hover_timer.start()
        super().showEvent(ev)

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()
        w = self.width()
        r = min(h / 2.0, 30.0)
        rect = QRectF(0.5, 0.5, w - 1.0, h - 1.0)
        g = QLinearGradient(0, 0, w, 0)
        g.setColorAt(0.0, QColor("#FFFFFF"))
        g.setColorAt(0.45, QColor("#F6F1FF"))
        g.setColorAt(1.0, QColor("#FFFFFF"))
        p.setBrush(QBrush(g))
        p.setPen(QPen(QColor(226, 213, 247, 235), 1.0))
        p.drawRoundedRect(rect, r, r)
        # soft purple aurora on the left edge (white/purple only), clearly drifting
        cx = 46.0 + 18.0 * math.sin(self._phase * 0.65)
        glow_a = 38 + int(18 * (0.5 + 0.5 * math.sin(self._phase)))
        aw = max(180.0, w * 0.20)
        rg = QLinearGradient(cx - aw, 0, cx + aw, 0)
        rg.setColorAt(0.0, QColor(124, 58, 237, 0))
        rg.setColorAt(0.5, QColor(124, 58, 237, glow_a))
        rg.setColorAt(1.0, QColor(124, 58, 237, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(rg))
        p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), r - 1, r - 1)
        # a visible shine band travelling across the capsule
        sweep = (math.sin(self._phase * 0.8) * 0.5 + 0.5)
        shine_x = w * (0.12 + 0.76 * sweep)
        sw = max(170.0, w * 0.17)
        shine = QLinearGradient(shine_x - sw, 0, shine_x + sw, 0)
        shine.setColorAt(0.0, QColor(196, 181, 253, 0))
        shine.setColorAt(0.5, QColor(196, 181, 253, 72))
        shine.setColorAt(1.0, QColor(196, 181, 253, 0))
        p.setBrush(QBrush(shine))
        p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), r - 1, r - 1)
        # pulsing border glow
        border_a = int(48 + 38 * (0.5 + 0.5 * math.sin(self._phase * 1.15)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(124, 58, 237, border_a), 1.35))
        p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), r - 1, r - 1)
        # a slow highlight sweep along the top edge
        sx = r + (w - 2 * r) * sweep
        p.setPen(QPen(QColor(124, 58, 237, 78), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(max(r, sx - 58), 1.8), QPointF(min(w - r, sx + 58), 1.8))
        # breathing glow behind the action chip on the right
        gx = max(40.0, w - 118.0)
        rg_a = int(9 + 11 * (0.5 + 0.5 * math.sin(self._phase * 1.05)))
        rig = QLinearGradient(gx - 110.0, 0, gx + 110.0, 0)
        rig.setColorAt(0.0, QColor(124, 58, 237, 0))
        rig.setColorAt(0.5, QColor(124, 58, 237, rg_a))
        rig.setColorAt(1.0, QColor(124, 58, 237, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(rig))
        p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), r - 1, r - 1)
        # drifting sparks and a counter-moving light on the lower edge
        for i in range(6):
            t = (self._phase * 0.045 + i * 0.19) % 1.0
            px = 26.0 + t * max(1.0, w - 52.0)
            py = h * (0.30 + 0.34 * math.sin(self._phase * 0.7 + i * 1.9))
            alpha = int(20 + 28 * (0.5 + 0.5 * math.sin(self._phase * 1.4 + i * 2.1)))
            rr = 1.3 + 0.7 * (i % 3)
            p.setBrush(QColor(124, 58, 237, alpha))
            p.drawEllipse(QPointF(px, py), rr, rr)
        bx = r + (w - 2 * r) * (1.0 - sweep)
        p.setPen(QPen(QColor(167, 139, 250, 64), 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(max(r, bx - 46), h - 1.8), QPointF(min(w - r, bx + 46), h - 1.8))


# ------------------------------------------------------------------ 通用视觉组件
class MiniSparkline(QWidget):
    def __init__(self, values: list[float], ma: list[float | None] | None = None, height: int = 116):
        super().__init__()
        self.values = [float(v) for v in values if v is not None]
        self.ma = ma or []
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._reveal = 0.0
        self._phase = 0.0
        _FRAME.subscribe(self._beat)
        self._reveal_anim = QVariantAnimation(self)
        self._reveal_anim.setDuration(900)
        self._reveal_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._reveal_anim.valueChanged.connect(self._reveal_step)
        QTimer.singleShot(120, self._start_reveal)

    def _beat(self, frame: int):
        if (frame & 1) or not self.isVisible():
            return
        self._phase = (self._phase + 0.050) % (math.pi * 2)
        self.update()

    def _reveal_step(self, value):
        self._reveal = float(value)
        self.update()

    def _start_reveal(self):
        self._reveal_anim.setStartValue(0.0)
        self._reveal_anim.setEndValue(1.0)
        self._reveal_anim.start()

    def paintEvent(self, e):  # noqa: N802
        if len(self.values) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        top, bottom, left, right = 10.0, h - 16.0, 2.0, w - 4.0
        lo, hi = min(self.values), max(self.values)
        for m in self.ma:
            if m is not None:
                lo, hi = min(lo, m), max(hi, m)
        if hi - lo < 1e-9:
            hi += 1
            lo -= 1
        pad = (hi - lo) * 0.12
        lo -= pad
        hi += pad
        n = len(self.values)

        def xy(i, v):
            return QPointF(left + (right - left) * i / (n - 1), top + (hi - v) / (hi - lo) * (bottom - top))

        # 网格
        grid_pen = QPen(QColor("#F1EDF8"), 1)
        p.setPen(grid_pen)
        for frac in (0.0, 0.5, 1.0):
            y = top + (bottom - top) * frac
            p.drawLine(QPointF(left, y), QPointF(right, y))

        pts = [xy(i, v) for i, v in enumerate(self.values)]
        line = QPainterPath(pts[0])
        for pt in pts[1:]:
            line.lineTo(pt)
        area = QPainterPath(line)
        area.lineTo(QPointF(pts[-1].x(), bottom))
        area.lineTo(QPointF(pts[0].x(), bottom))
        area.closeSubpath()
        k = max(0.0, min(1.0, self._reveal))
        p.save()
        p.setClipRect(QRectF(0, 0, w * (0.06 + 0.94 * k), h))
        grad = QLinearGradient(0, top, 0, bottom)
        grad.setColorAt(0, QColor(124, 58, 237, 52))
        grad.setColorAt(1, QColor(124, 58, 237, 0))
        p.fillPath(area, grad)
        p.setPen(QPen(QColor(C["purple"]), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(line)

        if self.ma and len(self.ma) == n:
            mpts = [xy(i, float(v)) for i, v in enumerate(self.ma) if v is not None]
            if len(mpts) > 1:
                mp = QPainterPath(mpts[0])
                for pt in mpts[1:]:
                    mp.lineTo(pt)
                pen = QPen(QColor("#B5A8CC"), 1.2, Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.drawPath(mp)
        p.restore()

        if k > 0.97 and len(pts) > 2:
            # a light tracer glides along the close-price line
            fi = (0.5 + 0.5 * math.sin(self._phase)) * (n - 1)
            i0 = min(n - 2, max(0, int(fi)))
            frac = fi - i0
            pt = QPointF(pts[i0].x() + (pts[i0 + 1].x() - pts[i0].x()) * frac,
                         pts[i0].y() + (pts[i0 + 1].y() - pts[i0].y()) * frac)
            trail = QPainterPath(pts[max(0, i0 - 7)])
            for j in range(max(0, i0 - 6), min(n, i0 + 1)):
                trail.lineTo(pts[j])
            trail.lineTo(pt)
            p.setPen(QPen(QColor(196, 181, 253, 140), 2.4, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPath(trail)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(124, 58, 237, 42))
            p.drawEllipse(pt, 8.5, 8.5)
            p.setBrush(QColor("#FFFFFF"))
            p.setPen(QPen(QColor(C["purple"]), 1.6))
            p.drawEllipse(pt, 3.0, 3.0)
        if k > 0.97:
            last = pts[-1]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(124, 58, 237, 35))
            p.drawEllipse(last, 7, 7)
            p.setBrush(QColor(C["purple"]))
            p.drawEllipse(last, 3.4, 3.4)
        p.end()


class ProbGauge(QWidget):
    def __init__(self, value: float, threshold: float = 0.55):
        super().__init__()
        self.value = max(0.0, min(1.0, value))
        self.threshold = threshold
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._disp = 0.0
        self._phase = 0.0
        _FRAME.subscribe(self._beat)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(850)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._step)
        QTimer.singleShot(100, self._start)

    def _beat(self, frame: int):
        if (frame & 1) or not self.isVisible():
            return
        self._phase = (self._phase + 0.064) % (math.pi * 2)
        self.update()

    def _step(self, value):
        self._disp = float(value)
        self.update()

    def _start(self):
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(self.value)
        self._anim.start()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = QRectF(0, 6, w, 12)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#F1ECFA"))
        p.drawRoundedRect(r, 6, 6)
        fill = QRectF(0, 6, max(12.0, w * self._disp), 12)
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0, QColor("#C4B5FD"))
        grad.setColorAt(0.55, QColor("#8B5CF6"))
        grad.setColorAt(1, QColor(C["purple"]))
        p.setBrush(grad)
        p.drawRoundedRect(fill, 6, 6)
        # a soft light band runs inside the filled part
        path = QPainterPath()
        path.addRoundedRect(fill, 6, 6)
        p.save()
        p.setClipPath(path)
        cx = -50.0 + (w + 100.0) * (0.5 + 0.5 * math.sin(self._phase))
        shine = QLinearGradient(cx - 34, 0, cx + 34, 0)
        shine.setColorAt(0.0, QColor(255, 255, 255, 0))
        shine.setColorAt(0.5, QColor(255, 255, 255, 96))
        shine.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(fill, shine)
        p.restore()
        x = w * self.threshold
        p.setPen(QPen(QColor(C["purple_d"]), 1.2, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x, 2), QPointF(x, h - 2))
        ex = w * self._disp
        breathe = 0.5 + 0.5 * math.sin(self._phase * 1.1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(124, 58, 237, int(26 + 22 * breathe)))
        p.drawEllipse(QPointF(ex, h / 2), 7.5 + 2.0 * breathe, 7.5 + 2.0 * breathe)
        p.setBrush(QColor(C["purple"]))
        p.drawEllipse(QPointF(ex, h / 2), 4.2, 4.2)
        p.end()


class ZoneBar(QWidget):
    def __init__(self, value: float):
        super().__init__()
        self.value = max(0.0, min(1.0, value))
        self.setFixedHeight(18)
        self._disp = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(850)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._step)
        QTimer.singleShot(100, self._start)

    def _step(self, value):
        self._disp = float(value)
        self.update()

    def _start(self):
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(self.value)
        self._anim.start()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        rect = QRectF(0, 5, w, 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#F1ECFA"))
        p.drawRoundedRect(rect, 4, 4)
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0, QColor("#EDE7F8"))
        grad.setColorAt(1, QColor("#C4B5FD"))
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(0, 5, max(8.0, w * self._disp), 8), 4, 4)
        x = max(6.0, min(w - 6.0, w * self._disp))
        p.setBrush(QColor("#FFFFFF"))
        p.setPen(QPen(QColor(C["purple"]), 2.4))
        p.drawEllipse(QPointF(x, h / 2), 5.2, 5.2)
        p.end()


class RateOrb(QWidget):
    """Review hit-rate ring with a breathing halo and orbiting comet."""

    def __init__(self, value: float, hits: int = 0, total: int = 0):
        super().__init__()
        self.value = max(0.0, min(1.0, float(value)))
        self.hits, self.total = int(hits), int(total)
        self.setFixedSize(92, 92)
        self._disp = 0.0
        self._phase = 0.0
        self._started = False
        _FRAME.subscribe(self._beat)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(900)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._step)
        self._fill_timer = QTimer(self)
        self._fill_timer.setSingleShot(True)
        self._fill_timer.setInterval(180)
        self._fill_timer.timeout.connect(self._start_fill)

    def _beat(self, frame: int):
        if (frame & 1) or not self.isVisible():
            return
        self._phase = (self._phase + 0.120) % (math.pi * 2)
        self.update()

    def _start_fill(self):
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(self.value)
        self._anim.start()

    def _step(self, value):
        self._disp = float(value)
        self.update()

    def showEvent(self, ev):  # noqa: N802
        if not self._started:
            self._started = True
            self._fill_timer.start()
        super().showEvent(ev)

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        c = QPointF(self.width() / 2.0, self.height() / 2.0)
        radius = (side - 20.0) / 2.0
        breathe = 0.5 + 0.5 * math.sin(self._phase)
        halo = QRadialGradient(c, radius + 12)
        halo.setColorAt(0.0, QColor(124, 58, 237, int(18 + 16 * breathe)))
        halo.setColorAt(0.7, QColor(124, 58, 237, 10))
        halo.setColorAt(1.0, QColor(124, 58, 237, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(c, radius + 12, radius + 12)
        # two gentle rings expand out of the orb and fade
        p.setBrush(Qt.BrushStyle.NoBrush)
        for off in (0.0, 0.5):
            t = (self._phase / (2 * math.pi) + off) % 1.0
            rr = radius + 2 + 8.0 * t
            p.setPen(QPen(QColor(124, 58, 237, int(44 * (1.0 - t))), 1.3))
            p.drawEllipse(c, rr, rr)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#FCFAFF"))
        p.setPen(QPen(QColor("#EFE8FA"), 1.0))
        p.drawEllipse(c, radius + 4, radius + 4)
        ring = QRectF(c.x() - radius, c.y() - radius, radius * 2, radius * 2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#F0EAF9"), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(ring, 90 * 16, -360 * 16)
        span = -int(360 * self._disp * 16)
        p.setPen(QPen(QColor(124, 58, 237, 38), 9.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(ring, 90 * 16, span)
        p.setPen(QPen(QColor(C["purple"]), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(ring, 90 * 16, span)
        if self._disp > 0.001:
            ang = math.radians(90 - 360 * self._disp)
            ex = c.x() + radius * math.cos(ang)
            ey = c.y() - radius * math.sin(ang)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(124, 58, 237, 42))
            p.drawEllipse(QPointF(ex, ey), 6.0, 6.0)
            p.setBrush(QColor("#FFFFFF"))
            p.setPen(QPen(QColor(C["purple"]), 1.8))
            p.drawEllipse(QPointF(ex, ey), 3.4, 3.4)
        # three small satellites circling the rim
        p.setPen(Qt.PenStyle.NoPen)
        for i, (da, alpha, rr) in enumerate(((0.0, 110, 2.3), (-0.9, 55, 1.7), (-1.8, 28, 1.3))):
            ang = self._phase * 0.85 + da
            ox = c.x() + (radius + 1.0) * math.cos(ang)
            oy = c.y() + (radius + 1.0) * math.sin(ang)
            p.setBrush(QColor(124, 58, 237, alpha))
            p.drawEllipse(QPointF(ox, oy), rr, rr)
            if i == 0:
                p.setBrush(QColor("#FFFFFF"))
                p.drawEllipse(QPointF(ox, oy), 0.9, 0.9)
        txt = f"{self.value * 100:.0f}%" if self.total else "—"
        p.setPen(QColor(C["purple_d"] if self.total else C["muted"]))
        p.setFont(_font(18, QFont.Weight.Bold))
        p.drawText(QRectF(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, txt)
        p.end()


class FloatingProgress(QWidget):
    """独立置顶的紫色悬浮进度球；主窗口最小化时仍留在桌面。"""

    SIZE = 104

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("TANDEM · 预测进度")
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.target = 0.0
        self.disp = 0.0
        self.stage = "准备"
        self.msg = ""
        self._pulse = 0.0
        self._last = time.perf_counter()
        self._drag_global: QPoint | None = None
        self._drag_pos: QPoint | None = None
        self._run_started = 0.0
        self._last_total = 720.0
        self._eta_value: float | None = None
        self._eta_at = 0.0
        self._pct = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def showEvent(self, ev):  # noqa: N802
        self._last = time.perf_counter()
        self._timer.start(16)
        super().showEvent(ev)

    def hideEvent(self, ev):  # noqa: N802
        self._timer.stop()
        super().hideEvent(ev)

    def _tick(self):
        now = time.perf_counter()
        dt = min(0.05, max(0.0, now - self._last))
        self._last = now
        diff = self.target - self.disp
        if abs(diff) > 0.01:
            k = 1.0 - math.exp(-dt * 9.0)
            self.disp += diff * k
            if abs(self.target - self.disp) < 0.01:
                self.disp = self.target
        self._pulse = (self._pulse + dt * 2.4) % (math.pi * 2)
        self.update()

    def _estimate_eta(self, pct: float, now: float) -> None:
        if pct >= 99.5:
            self._eta_value = 0.0
            self._eta_at = now
            return
        if self._run_started <= 0:
            return
        elapsed = max(0.0, now - self._run_started)
        base = max(0.0, self._last_total - elapsed)
        if pct >= 3.0:
            naive = elapsed * (100.0 - pct) / max(pct, 0.5)
            w = min(0.75, max(0.15, pct / 100.0))
            target = base * (1.0 - w) + naive * w
        else:
            target = base
        target = max(15.0, min(3600.0, target))
        if self._eta_value is None:
            self._eta_value = target
        else:
            self._eta_value = 0.25 * self._eta_value + 0.75 * target
        self._eta_at = now

    def _remaining_text(self) -> str:
        if self.target >= 99.5:
            return "即将完成"
        if self._eta_value is None:
            return "估算中…"
        remain = max(0.0, self._eta_value - max(0.0, time.perf_counter() - self._eta_at))
        if remain < 30:
            return "还剩 <1 分钟"
        minutes = int(round(remain / 60.0))
        if minutes >= 60:
            return "还剩 60+ 分钟"
        return f"还剩 {max(1, minutes)} 分钟"

    def begin(self, anchor=None):
        self.target = 0.0
        self.disp = 0.0
        self.stage = "准备"
        self._pct = 0.0
        latest = _read_json(LATEST)
        digits = "".join(ch for ch in str(latest.get("耗时", "") or "") if ch.isdigit())
        self._last_total = 720.0
        if digits:
            try:
                self._last_total = min(3600.0, max(300.0, float(digits)))
            except ValueError:
                pass
        now = time.perf_counter()
        self._run_started = now
        self._eta_at = now
        self._eta_value = self._last_total
        self.msg = f"参考上次全程约 {self._last_total / 60:.0f} 分钟"
        rect = anchor if anchor is not None else QApplication.primaryScreen().availableGeometry()
        self.move(rect.right() - self.width() - 24, rect.bottom() - self.height() - 32)
        self.setToolTip(self.stage + "｜" + self.msg + "｜" + self._remaining_text())
        self.show()
        self.raise_()

    def update_progress(self, d: dict):
        try:
            pct = float(d.get("pct", 0))
        except (TypeError, ValueError):
            pct = 0.0
        pct = max(0.0, min(100.0, pct))
        self.target = max(self.target, pct)
        self.stage = str(d.get("stage") or "运行中")
        self.msg = str(d.get("msg") or "")
        if pct > self._pct + 0.01:
            self._pct = pct
            self._estimate_eta(pct, time.perf_counter())
        tip = f"{self.stage}｜{self.msg}" if self.msg else self.stage
        self.setToolTip(f"{tip}｜{self._remaining_text()}")
        self.update()

    def finish(self):
        self.target = 100.0
        self.stage = "完成"
        self.msg = "结果已刷新"
        self._eta_value = 0.0
        self._eta_at = time.perf_counter()
        self.setToolTip("预测完成｜结果已刷新")
        QTimer.singleShot(1500, self._hide_later)

    def _hide_later(self):
        if self.target >= 99:
            self.hide()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        c = QPointF(w / 2, h / 2)
        radius = (min(w, h) - 16) / 2
        # 外光晕
        halo = QRadialGradient(c, radius + 12)
        halo.setColorAt(0, QColor(124, 58, 237, int(18 + 10 * (1 + math.sin(self._pulse)))))
        halo.setColorAt(0.7, QColor(124, 58, 237, 10))
        halo.setColorAt(1, QColor(124, 58, 237, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(c, radius + 13, radius + 13)
        # 球体
        p.setBrush(QColor(255, 255, 255, 240))
        p.setPen(QPen(QColor(231, 220, 246), 1))
        p.drawEllipse(c, radius + 3, radius + 3)
        # 轨道
        ring = QRectF(c.x() - radius, c.y() - radius, radius * 2, radius * 2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#F0EAF9"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(ring, 90 * 16, -360 * 16)
        # 进度弧 + 光晕
        span = -int(360 * self.disp / 100 * 16)
        p.setPen(QPen(QColor(124, 58, 237, 36), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(ring, 90 * 16, span)
        p.setPen(QPen(QColor(C["purple"]), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(ring, 90 * 16, span)
        # 进度端点
        ang = math.radians(90 - 360 * self.disp / 100)
        ex, ey = c.x() + radius * math.cos(ang), c.y() - radius * math.sin(ang)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(C["purple"]))
        p.drawEllipse(QPointF(ex, ey), 3.2, 3.2)
        # 环绕小卫星：让静止的进度球也有连续、低成本的动画
        oa = self._pulse * 0.9
        for k, (da, ar, rr) in enumerate(((0.0, 95, 5.0), (-0.18, 45, 3.6), (-0.36, 20, 2.4))):
            ang = oa + da
            ox = c.x() + (radius + 1.5) * math.cos(ang)
            oy = c.y() + (radius + 1.5) * math.sin(ang)
            p.setBrush(QColor(124, 58, 237, ar))
            p.drawEllipse(QPointF(ox, oy), rr, rr)
            if k == 0:
                p.setBrush(QColor("#FFFFFF"))
                p.drawEllipse(QPointF(ox, oy), 1.8, 1.8)
        # 百分比、剩余时间与状态
        p.setPen(QColor(C["purple_d"]))
        p.setFont(_font(17, QFont.Weight.Bold))
        p.drawText(QRectF(8, 21, w - 16, 27), Qt.AlignmentFlag.AlignCenter, f"{self.disp:.1f}%")
        p.setPen(QColor(C["purple_d"]))
        p.setFont(_font(9.5, QFont.Weight.DemiBold, cn=True))
        p.drawText(QRectF(4, 47, w - 8, 15), Qt.AlignmentFlag.AlignCenter, self._remaining_text())
        stage = self.stage or "运行中"
        if len(stage) > 6:
            stage = stage[:6] + "…"
        p.setPen(QColor(C["muted"]))
        p.setFont(_font(7.5, QFont.Weight.DemiBold, cn=True))
        p.drawText(QRectF(6, 62, w - 12, 13), Qt.AlignmentFlag.AlignCenter, stage)
        p.end()

    def mousePressEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_global = e.globalPosition().toPoint()
            self._drag_pos = self.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):  # noqa: N802
        if self._drag_global is not None and self._drag_pos is not None and e.buttons() & Qt.MouseButton.LeftButton:
            delta = e.globalPosition().toPoint() - self._drag_global
            pos = self._drag_pos + delta
            screen = self.screen() or QApplication.primaryScreen()
            if screen is not None:
                g = screen.availableGeometry()
                pos.setX(max(g.left() + 8, min(pos.x(), g.right() - self.width() - 8)))
                pos.setY(max(g.top() + 8, min(pos.y(), g.bottom() - self.height() - 8)))
            self.move(pos)

    def mouseReleaseEvent(self, e):  # noqa: N802
        self._drag_global = None
        self._drag_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)


# ------------------------------------------------------------------ 布局组件
class ReasonRow(QFrame):
    """Reason/trigger row with a cursor-tracked hover wash and left accent."""

    def __init__(self, delay: int = 0):
        super().__init__()
        self.setObjectName("ReasonRow")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._t = 0.0
        self._inside = False
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._step)
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._check)
        self._reveal = 0.0
        self._reveal_started = False
        self._delay = 110 + max(0, int(delay))
        self._reveal_anim = QVariantAnimation(self)
        self._reveal_anim.setDuration(620)
        self._reveal_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._reveal_anim.valueChanged.connect(self._reveal_step)
        self._glint = 1.0
        self._glint_anim = QVariantAnimation(self)
        self._glint_anim.setDuration(620)
        self._glint_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._glint_anim.valueChanged.connect(self._glint_step)
        self._reveal_timer = QTimer(self)
        self._reveal_timer.setSingleShot(True)
        self._reveal_timer.setInterval(self._delay)
        self._reveal_timer.timeout.connect(self._start_reveal)

    def _step(self, value):
        self._t = float(value)
        self.update()

    def _to(self, target):
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(target)
        self._anim.start()

    def _check(self):
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        if inside != self._inside:
            self._inside = inside
            self._to(1.0 if inside else 0.0)
            if inside:
                self._start_glint()

    def _start_reveal(self):
        self._reveal_anim.stop()
        self._reveal_anim.setStartValue(0.0)
        self._reveal_anim.setEndValue(1.0)
        self._reveal_anim.start()

    def _reveal_step(self, value):
        self._reveal = float(value)
        self.update()

    def _start_glint(self):
        self._glint_anim.stop()
        self._glint_anim.setStartValue(0.0)
        self._glint_anim.setEndValue(1.0)
        self._glint_anim.start()

    def _glint_step(self, value):
        self._glint = float(value)
        self.update()

    def showEvent(self, ev):  # noqa: N802
        self._timer.start()
        if not self._reveal_started:
            self._reveal_started = True
            self._reveal_timer.start()
        super().showEvent(ev)

    def hideEvent(self, ev):  # noqa: N802
        self._timer.stop()
        super().hideEvent(ev)

    def paintEvent(self, ev):  # noqa: N802
        super().paintEvent(ev)
        revealing = self._reveal < 0.999
        hovering = self._t > 0.01
        glinting = self._glint < 0.999
        if not (revealing or hovering or glinting):
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        if revealing:
            e = 1.0 - (1.0 - self._reveal) ** 3
            if e > 0.01:
                bar_h = max(2.0, (h - 14.0) * e)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(124, 58, 237, int(92 * e)))
                p.drawRoundedRect(QRectF(0, (h - bar_h) / 2.0, 2.6, bar_h), 1.3, 1.3)
                cx = -70.0 + (w + 140.0) * e
                g = QLinearGradient(cx - 60, 0, cx + 60, 0)
                g.setColorAt(0.0, QColor(196, 181, 253, 0))
                g.setColorAt(0.5, QColor(196, 181, 253, int(34 * (1.0 - e) + 8)))
                g.setColorAt(1.0, QColor(196, 181, 253, 0))
                p.fillRect(QRectF(0, 0, w, h), g)
        if hovering:
            e = _spring(self._t, response=0.55, damping=0.86)
            e = min(1.0, e)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(124, 58, 237, int(16 * e)))
            p.drawRoundedRect(QRectF(0, 0, w, h), 9, 9)
            p.setBrush(QColor(124, 58, 237, int(205 * e)))
            p.drawRoundedRect(QRectF(0, 7, 3, max(2.0, h - 14)), 1.5, 1.5)
        if glinting:
            ge = 1.0 - (1.0 - self._glint) ** 3
            cx = -80.0 + (w + 160.0) * ge
            g = QLinearGradient(cx - 55, 0, cx + 55, 0)
            g.setColorAt(0.0, QColor(124, 58, 237, 0))
            g.setColorAt(0.5, QColor(124, 58, 237, int(44 * (1.0 - ge))))
            g.setColorAt(1.0, QColor(124, 58, 237, 0))
            p.fillRect(QRectF(1, 0, max(0.0, w - 2), h), g)
        p.end()


class TableRowHighlight(QWidget):
    """Painted hover row for the read-only candidate table."""

    def __init__(self, table):
        super().__init__(table.viewport())
        self._table = table
        self._t = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._step)
        self._row = -1
        self._glint = 1.0
        self._glint_anim = QVariantAnimation(self)
        self._glint_anim.setDuration(560)
        self._glint_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._glint_anim.valueChanged.connect(self._glint_step)
        self.hide()

    def _step(self, value):
        self._t = float(value)
        self.update()
        if self._t <= 0.01 and not self.isHidden():
            self.hide()

    def _glint_step(self, value):
        self._glint = float(value)
        self.update()

    def _to(self, target):
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(target)
        self._anim.start()

    def show_row(self, row: int):
        if row < 0:
            self._row = -1
            self._to(0.0)
            return
        rect = self._table.visualRect(self._table.model().index(row, 0))
        if not rect.isValid():
            self._row = -1
            self._to(0.0)
            return
        if row != self._row:
            self._row = row
            self._glint = 0.0
            self._glint_anim.stop()
            self._glint_anim.setStartValue(0.0)
            self._glint_anim.setEndValue(1.0)
            self._glint_anim.start()
        self.setGeometry(0, rect.y(), self._table.viewport().width(), rect.height())
        self.show()
        self.raise_()
        self._to(1.0)

    def hide_row(self):
        self._row = -1
        self._to(0.0)

    def paintEvent(self, ev):  # noqa: N802
        if self._t <= 0.0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        e = _spring(self._t, response=0.55, damping=0.86)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(124, 58, 237, int(18 * min(1.0, e))))
        p.drawRoundedRect(QRectF(0, 1, self.width(), self.height() - 2), 6, 6)
        p.setBrush(QColor(124, 58, 237, int(190 * min(1.0, e))))
        p.drawRoundedRect(QRectF(0, 5, 3, max(2.0, self.height() - 10)), 1.5, 1.5)
        if self._glint < 0.999:
            ge = 1.0 - (1.0 - self._glint) ** 3
            cx = -60.0 + (self.width() + 120.0) * ge
            g = QLinearGradient(cx - 45, 0, cx + 45, 0)
            g.setColorAt(0.0, QColor(124, 58, 237, 0))
            g.setColorAt(0.5, QColor(124, 58, 237, int(40 * (1.0 - ge))))
            g.setColorAt(1.0, QColor(124, 58, 237, 0))
            p.fillRect(QRectF(0, 1, self.width(), self.height() - 2), g)
        p.end()


class TableHoverFilter(QObject):
    def __init__(self, table, highlight):
        super().__init__(table)
        self._table = table
        self._hl = highlight

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.MouseMove:
            row = self._table.rowAt(int(ev.position().y()))
            if row >= 0:
                self._hl.show_row(row)
        elif ev.type() == QEvent.Type.Leave:
            self._hl.hide_row()
        return False


class HeaderSheen(QWidget):
    """Transparent travelling highlight over the candidate table header."""

    def __init__(self, host: QWidget):
        super().__init__(host)
        self._host = host
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._phase = 0.0
        host.installEventFilter(self)
        self.setGeometry(host.rect())
        self.show()
        _FRAME.subscribe(self._beat)

    def _beat(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.040) % (math.pi * 2)
        self.update()

    def eventFilter(self, obj, ev):  # noqa: N802
        if obj is self._host and ev.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.setGeometry(self._host.rect())
            self.raise_()
        return False

    def paintEvent(self, ev):  # noqa: N802
        if self.width() < 4 or self.height() < 4:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        band = 120.0
        cx = -band + (w + 2 * band) * (0.5 + 0.5 * math.sin(self._phase * 0.82))
        g = QLinearGradient(cx - band, 0, cx + band, 0)
        g.setColorAt(0.0, QColor(255, 255, 255, 0))
        g.setColorAt(0.5, QColor(255, 255, 255, 44))
        g.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(QRectF(0, 0, w, h), g)
        p.setPen(QPen(QColor(196, 181, 253, int(60 + 50 * (0.5 + 0.5 * math.sin(self._phase * 1.3)))), 1.0))
        p.drawLine(QPointF(0, h - 1), QPointF(w, h - 1))
        p.end()


class TableAura(QWidget):
    """Breathing glow on the top candidate plus a one-shot vertical sweep."""

    def __init__(self, table):
        super().__init__(table.viewport())
        self._table = table
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._phase = 0.0
        self._sweep = 0.0
        self._sweep_started = False
        table.viewport().installEventFilter(self)
        self._sweep_anim = QVariantAnimation(self)
        self._sweep_anim.setDuration(950)
        self._sweep_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._sweep_anim.valueChanged.connect(self._sweep_step)
        self.setGeometry(table.viewport().rect())
        self.show()
        _FRAME.subscribe(self._beat)

    def _row_h(self) -> int:
        try:
            return int(self._table.rowHeight(0) or 31)
        except Exception:  # noqa: BLE001
            return 31

    def _beat(self, frame: int):
        if (frame & 1) or not self.isVisible():
            return
        self._phase = (self._phase + 0.060) % (math.pi * 2)
        if self._sweep < 0.999:
            self.update()
        else:
            self.update(QRect(0, 0, max(1, self.width()), self._row_h() + 2))

    def _sweep_step(self, value):
        self._sweep = float(value)
        self.update()

    def eventFilter(self, obj, ev):  # noqa: N802
        if obj is self._table.viewport() and ev.type() == QEvent.Type.Resize:
            self.setGeometry(self._table.viewport().rect())
            self.raise_()
        return False

    def showEvent(self, ev):  # noqa: N802
        self.setGeometry(self._table.viewport().rect())
        self.raise_()
        if not self._sweep_started:
            self._sweep_started = True
            self._sweep_anim.setStartValue(0.0)
            self._sweep_anim.setEndValue(1.0)
            self._sweep_anim.start()
        super().showEvent(ev)

    def paintEvent(self, ev):  # noqa: N802
        if self._table.rowCount() < 1 or self.width() < 4 or self.height() < 4:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        row_h = float(self._row_h())
        breathe = 0.5 + 0.5 * math.sin(self._phase)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(124, 58, 237, int(7 + 9 * breathe)))
        p.drawRoundedRect(QRectF(0, 1, w - 2, max(4.0, row_h - 2)), 6, 6)
        p.setBrush(QColor(124, 58, 237, int(120 + 65 * breathe)))
        p.drawRoundedRect(QRectF(0, 5, 3, max(2.0, row_h - 10)), 1.5, 1.5)
        # a shine drifts along the leader row
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 1, w - 2, max(4.0, row_h - 2)), 6, 6)
        p.save()
        p.setClipPath(path)
        cx = -70.0 + (w + 140.0) * (0.5 + 0.5 * math.sin(self._phase * 0.65))
        g = QLinearGradient(cx - 50, 0, cx + 50, 0)
        g.setColorAt(0.0, QColor(124, 58, 237, 0))
        g.setColorAt(0.5, QColor(124, 58, 237, 32))
        g.setColorAt(1.0, QColor(124, 58, 237, 0))
        p.fillRect(QRectF(0, 0, w, row_h), g)
        p.restore()
        if self._sweep < 0.999:
            e = 1.0 - (1.0 - self._sweep) ** 3
            y = -30.0 + (h + 60.0) * e
            sg = QLinearGradient(0, y - 32, 0, y + 32)
            sg.setColorAt(0.0, QColor(196, 181, 253, 0))
            sg.setColorAt(0.5, QColor(196, 181, 253, 38))
            sg.setColorAt(1.0, QColor(196, 181, 253, 0))
            p.fillRect(QRectF(0, max(0.0, y - 32), w, min(64.0, h)), sg)
        p.end()


class ReasonList(QWidget):
    def __init__(self, items: list[str], compact: bool = False):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        for i, text in enumerate(items):
            row = ReasonRow(delay=i * 55)
            h = QHBoxLayout(row)
            h.setContentsMargins(14, 10, 14, 10)
            h.setSpacing(10)
            body = _label(text, _font(12 if compact else 12.5, cn=True), C["ink2"], wrap=True)
            body.setStyleSheet("color:#443D50;background:transparent;line-height:150%;")
            h.addWidget(body, 1)
            lay.addWidget(row)


class StatTile(QFrame):
    def __init__(self, key: str, value: str, sub: str = "", accent: bool = False):
        super().__init__()
        self.setObjectName("Tile")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(13, 10, 13, 11)
        lay.setSpacing(3)
        k = _label(key, _font(11, cn=True), C["muted"])
        v = _label(value, _font(19, QFont.Weight.Bold), C["purple_d"] if accent else C["ink"])
        lay.addWidget(k)
        lay.addWidget(v)
        if sub:
            lay.addWidget(_label(sub, _font(10.5, cn=True), C["muted"]))
        self._accent = accent
        self._t = 0.0
        self._inside = False
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(260)
        self._anim.valueChanged.connect(self._step)
        self.installEventFilter(self)
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(50)
        self._hover_timer.timeout.connect(self._check_hover)

    def _check_hover(self):
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        if inside != self._inside:
            self._inside = inside
            self._anim.stop()
            self._anim.setStartValue(self._t)
            self._anim.setEndValue(1.0 if inside else 0.0)
            self._anim.start()

    def showEvent(self, ev):  # noqa: N802
        self._hover_timer.start()
        super().showEvent(ev)

    def hideEvent(self, ev):  # noqa: N802
        self._hover_timer.stop()
        super().hideEvent(ev)

    def _step(self, value) -> None:
        self._t = float(value)
        self.update()

    def eventFilter(self, obj, ev):  # noqa: N802
        if ev.type() == QEvent.Type.Enter:
            self._anim.stop()
            self._anim.setStartValue(self._t)
            self._anim.setEndValue(1.0)
            self._anim.start()
        elif ev.type() == QEvent.Type.Leave:
            self._anim.stop()
            self._anim.setStartValue(self._t)
            self._anim.setEndValue(0.0)
            self._anim.start()
        return False

    def paintEvent(self, ev):  # noqa: N802
        super().paintEvent(ev)
        if self._t <= 0.01:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        e = _spring(self._t, response=0.55, damping=0.8)
        w = (self.width() - 26) * min(1.0, e)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(124, 58, 237, 210))
        p.drawRoundedRect(QRectF(13, self.height() - 5.5, max(2.0, w), 2.4), 1.2, 1.2)


class Card(QFrame):
    """Card with a cheap painted hover rim (no drop-shadow effect)."""

    def __init__(self):
        super().__init__()
        self.setObjectName("Card")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._t = 0.0
        self._inside = False
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._step)
        self._hover_timer = QTimer(self)
        self._hover_timer.setInterval(50)
        self._hover_timer.timeout.connect(self._check_hover)
        self._phase = (id(self) % 628) / 100.0
        _FRAME.subscribe(self._breathe)

    def _breathe(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.030) % (math.pi * 2)
        # only the top strip changes; keep the large card surface untouched
        self.update(QRect(0, 0, max(1, self.width()), 5))

    def _step(self, value):
        self._t = float(value)
        self.update()

    def _to(self, target):
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(target)
        self._anim.start()

    def _check_hover(self):
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        if inside != self._inside:
            self._inside = inside
            self._to(1.0 if inside else 0.0)

    def showEvent(self, ev):  # noqa: N802
        self._hover_timer.start()
        super().showEvent(ev)

    def hideEvent(self, ev):  # noqa: N802
        self._hover_timer.stop()
        super().hideEvent(ev)

    def paintEvent(self, ev):  # noqa: N802
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        if w < 4 or h < 4:
            p.end()
            return
        rect = QRectF(1, 1, w - 2, h - 2)
        # slow breathing rim
        breathe = 0.5 + 0.5 * math.sin(self._phase)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(124, 58, 237, int(6 + 6 * breathe)), 1.0))
        p.drawRoundedRect(rect, 19, 19)
        # travelling light along the top rim
        path = QPainterPath()
        path.addRoundedRect(rect, 19, 19)
        p.save()
        p.setClipPath(path)
        cx = w * (0.5 + 0.5 * math.sin(self._phase * 0.62))
        g = QLinearGradient(cx - 130, 0, cx + 130, 0)
        g.setColorAt(0.0, QColor(167, 139, 250, 0))
        g.setColorAt(0.5, QColor(167, 139, 250, 64))
        g.setColorAt(1.0, QColor(167, 139, 250, 0))
        p.fillRect(QRectF(0, 0, w, 1.6), g)
        p.restore()
        if self._t > 0.01:
            e = _spring(self._t, response=0.55, damping=0.86)
            p.setPen(QPen(QColor(124, 58, 237, int(95 * min(1.0, e))), 1.5))
            p.drawRoundedRect(rect, 19, 19)
        p.end()


class GlowRail(QFrame):
    """Island rail with a slow lavender light moving across it."""

    def __init__(self):
        super().__init__()
        self.setObjectName("IslandRail")
        self.setStyleSheet(
            "QFrame#IslandRail{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #FFFFFF,stop:1 #F7F2FF);border:1px solid #E7DDF7;border-radius:18px;}"
        )
        self._phase = 0.0
        _FRAME.subscribe(self._beat)

    def _beat(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.022) % (math.pi * 2)
        self.update()

    def paintEvent(self, ev):  # noqa: N802
        super().paintEvent(ev)
        w, h = float(self.width()), float(self.height())
        if w < 8 or h < 8:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 18, 18)
        p.setClipPath(path)
        cx = -140.0 + (w + 280.0) * (0.5 + 0.5 * math.sin(self._phase * 0.55))
        g = QLinearGradient(cx - 120, 0, cx + 120, 0)
        g.setColorAt(0.0, QColor(167, 139, 250, 0))
        g.setColorAt(0.5, QColor(167, 139, 250, 36))
        g.setColorAt(1.0, QColor(167, 139, 250, 0))
        p.fillRect(QRectF(0, 0, w, h), g)
        p.setPen(QPen(QColor(167, 139, 250, 75), 1.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(max(0.0, cx - 45), h - 1.2), QPointF(min(w, cx + 45), h - 1.2))
        p.end()


def _card() -> tuple[QFrame, QVBoxLayout]:
    card = Card()
    lay = QVBoxLayout(card)
    lay.setContentsMargins(24, 22, 24, 22)
    lay.setSpacing(14)
    return card, lay


def _card_head(letter: str, title: str, sub: str, right: QWidget | None = None, letter_bg: str = "#F0E9FF", letter_fg: str = "#6D28D9") -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(11)
    badge = HoverBadge(letter, 34, 34, 11, letter_bg, letter_fg, 15)
    col = QVBoxLayout()
    col.setSpacing(1)
    title_lbl = _label(title, _font(15.5, QFont.Weight.Bold, cn=True), C["ink"])
    col.addWidget(title_lbl)
    row.addWidget(badge)
    row.addLayout(col)
    row.addStretch(1)
    if right is not None:
        row.addWidget(right, 0, Qt.AlignmentFlag.AlignTop)
    return row


def _hline() -> QFrame:
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet("background:#F0EBF7;border:none;")
    return line


def _section_title(text: str) -> QLabel:
    lab = _label(text, _font(12, QFont.Weight.Bold, cn=True), C["purple_d"])
    lab.setStyleSheet("color:#6D28D9;background:transparent;letter-spacing:0.5px;")
    return lab


def _section_bar(num: str, title: str, sub: str) -> QWidget:
    box = QWidget()
    box.setObjectName("SectionBar")
    lay = QHBoxLayout(box)
    lay.setContentsMargins(4, 8, 4, 0)
    lay.setSpacing(10)
    lay.addWidget(PulseDot(6), 0, Qt.AlignmentFlag.AlignVCenter)
    title_lbl = _label(title, _font(14, QFont.Weight.Bold, cn=True), C["ink"])
    lay.addWidget(title_lbl)
    lay.addWidget(SectionLine(), 1)
    return box


def _primary_card() -> tuple[QFrame, QVBoxLayout]:
    card, lay = _card()
    lay.addWidget(AccentBar())
    return card, lay


class IconBadge(QLabel):
    def __init__(self, size: int = 32):
        super().__init__()
        self.setFixedSize(size, size)
        self.setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #F1EAFF,stop:1 #E6DBFF);border-radius:10px;")
        pm = QPixmap(str(ASSETS / "app.png"))
        if not pm.isNull():
            self.setPixmap(pm.scaled(size - 8, size - 8, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class IndustryRow(QFrame):
    """One SW2 industry row; hover drives the member popup and a soft wash."""

    def __init__(self, on_enter, on_leave):
        super().__init__()
        self.setFixedHeight(28)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setStyleSheet("background:transparent;border:none;")
        self._on_enter = on_enter
        self._on_leave = on_leave
        self._item: dict = {}
        self._pct: float | None = None
        self._val = 0.0
        self._max_abs = 1.0
        self._t = 0.0
        self._bar = 0.0
        self._leader = False
        self._phase = (id(self) % 628) / 100.0
        _FRAME.subscribe(self._beat)
        self._hover_anim = QVariantAnimation(self)
        self._hover_anim.setDuration(180)
        self._hover_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._hover_anim.valueChanged.connect(self._step)
        self._bar_anim = QVariantAnimation(self)
        self._bar_anim.setDuration(620)
        self._bar_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._bar_anim.valueChanged.connect(self._bar_step)
        self._val_anim = QVariantAnimation(self)
        self._val_anim.setDuration(780)
        self._val_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._val_anim.valueChanged.connect(self._val_step)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(5, 0, 6, 0)
        lay.setSpacing(6)
        self.rank = _label("", _font(9.5), C["muted"])
        self.rank.setFixedWidth(15)
        self.rank.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.name = _label("", _font(11.5, QFont.Weight.DemiBold, cn=True), C["ink2"])
        self.name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.pct = _label("—", _font(11.5, QFont.Weight.Bold), C["muted"])
        self.pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.pct.setMinimumWidth(58)
        self.pct.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lay.addWidget(self.rank)
        lay.addWidget(self.name, 1)
        lay.addWidget(self.pct)

    def _step(self, value):
        self._t = float(value)
        self.update()

    def _bar_step(self, value):
        self._bar = float(value)
        self.update()

    def _val_step(self, value):
        self._val = float(value)
        self.pct.setText(f"{self._val * 100:+.2f}%")
        self.update()

    def _beat(self, frame: int):
        if not self._leader or not self.isVisible():
            return
        self._phase = (self._phase + 0.035) % (math.pi * 2)
        self.update()

    def set_leader(self, enabled: bool) -> None:
        self._leader = bool(enabled)
        self.update()

    def set_data(self, rank: int, item: dict, max_abs: float) -> None:
        self._item = dict(item)
        self._pct = item.get("pct")
        self._max_abs = max(1e-8, abs(float(max_abs or 0.0)))
        self.rank.setText(str(rank))
        self.name.setText(_display_industry(str(item.get("name") or "—")))
        if self._pct is None:
            self._val_anim.stop()
            self._val = 0.0
            self.pct.setText("—")
            self.pct.setStyleSheet(f"color:{C['muted']};background:transparent;")
            frac = 0.0
        else:
            val = float(self._pct)
            self.pct.setStyleSheet(
                f"color:{'#6D28D9' if val > 0 else (C['muted'] if val == 0 else '#8A8794')};background:transparent;"
            )
            self._val_anim.stop()
            self._val_anim.setStartValue(self._val)
            self._val_anim.setEndValue(val)
            self._val_anim.start()
            frac = min(1.0, abs(val) / self._max_abs)
        self._bar_anim.stop()
        self._bar_anim.setStartValue(self._bar)
        self._bar_anim.setEndValue(frac)
        self._bar_anim.start()

    def enterEvent(self, ev):  # noqa: N802
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self._t)
        self._hover_anim.setEndValue(1.0)
        self._hover_anim.start()
        self._on_enter(self)
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self._t)
        self._hover_anim.setEndValue(0.0)
        self._hover_anim.start()
        self._on_leave(self)
        super().leaveEvent(ev)

    def paintEvent(self, ev):  # noqa: N802
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        e = _spring(self._t, response=0.42, damping=0.86)
        if self._leader:
            breathe = 0.5 + 0.5 * math.sin(self._phase)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(124, 58, 237, int(7 + 7 * breathe)))
            p.drawRoundedRect(QRectF(0, 1, w, h - 2), 7, 7)
            cx = w * (0.5 + 0.5 * math.sin(self._phase * 0.65))
            g = QLinearGradient(cx - 70, 0, cx + 70, 0)
            g.setColorAt(0.0, QColor(167, 139, 250, 0))
            g.setColorAt(0.5, QColor(167, 139, 250, 34))
            g.setColorAt(1.0, QColor(167, 139, 250, 0))
            p.setBrush(QBrush(g))
            p.drawRoundedRect(QRectF(0, 1, w, h - 2), 7, 7)
        if e > 0.01:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(124, 58, 237, int(15 * min(1.0, e))))
            p.drawRoundedRect(QRectF(0, 1, w, h - 2), 7, 7)
            p.setBrush(QColor(124, 58, 237, int(185 * min(1.0, e))))
            p.drawRoundedRect(QRectF(0, 4, 2.5, max(2.0, h - 8)), 1.25, 1.25)
        if self._bar > 0.001:
            color = QColor(124, 58, 237, 105) if (self._pct or 0) > 0 else QColor(141, 134, 154, 70)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(5, h - 2.2, max(2.0, (w - 11) * self._bar), 1.6), 0.8, 0.8)
        p.end()


class IndustryPopup(QWidget):
    """Floating member list anchored to the hovered industry row."""

    W = 272

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self.setFixedWidth(self.W)
        self.resize(self.W, 120)
        self.current_code = ""
        self._name = ""
        self._pct: float | None = None
        self._stocks: list[dict] = []
        self._loading = False
        self._t = 0.0
        self._phase = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(190)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._step)
        _FRAME.subscribe(self._beat)
        self.hide()

    def _step(self, value):
        self._t = float(value)
        self.update()

    def _beat(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.045) % (math.pi * 2)
        self.update()

    def _calc_height(self) -> int:
        n = min(10, len(self._stocks))
        if not n:
            n = 1
        return 44 + n * 21 + (10 if len(self._stocks) > 10 else 0) + 10

    def _place(self, anchor: QWidget) -> None:
        host = self.parentWidget()
        if host is None:
            return
        pos = anchor.mapTo(host, QPoint(0, 0))
        x = pos.x() + anchor.width() + 54
        if x + self.width() > host.width() - 10:
            x = pos.x() - self.width() - 14
        x = max(10, min(x, host.width() - self.width() - 10))
        y = max(10, min(pos.y() - 8, host.height() - self.height() - 10))
        self.move(x, y)

    def show_for(self, code: str, name: str, pct: float | None, stocks: list[dict],
                 anchor: QWidget, loading: bool) -> None:
        self.current_code = code
        self._name = name
        self._pct = pct
        self._stocks = stocks
        self._loading = loading
        self.resize(self.W, self._calc_height())
        self._place(anchor)
        was_visible = self.isVisible()
        self.show()
        self.raise_()
        QTimer.singleShot(0, self.raise_)
        if not was_visible or self._t < 0.99:
            self._anim.stop()
            self._anim.setStartValue(self._t)
            self._anim.setEndValue(1.0)
            self._anim.start()

    def update_stocks(self, stocks: list[dict], loading: bool) -> None:
        if not self.isVisible():
            return
        self._stocks = stocks
        self._loading = loading
        self.resize(self.W, self._calc_height())
        self.update()

    def hide_popup(self) -> None:
        if not self.isVisible():
            return
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self._hide_if_zero)
        self._anim.start()

    def hide_now(self) -> None:
        self._anim.stop()
        self._t = 0.0
        self.current_code = ""
        self.hide()

    def _hide_if_zero(self) -> None:
        if self._t <= 0.02:
            self.hide()

    def paintEvent(self, ev):  # noqa: N802
        e = _spring(min(1.0, max(0.0, self._t)), response=0.42, damping=0.9)
        if e <= 0.001:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(min(1.0, max(0.0, e)))
        p.translate(0, (1.0 - min(1.0, e)) * 7.0)
        w, h = float(self.width()), float(self.height())
        rect = QRectF(1, 1, w - 2, h - 2)
        path = QPainterPath()
        path.addRoundedRect(rect, 16, 16)
        p.fillPath(path, QColor("#FFFFFF"))
        p.setPen(QPen(QColor("#E5DAF6"), 1.0))
        p.drawPath(path)
        p.save()
        p.setClipPath(path)
        hg = QLinearGradient(0, 0, w, 44)
        hg.setColorAt(0.0, QColor("#F7F2FF"))
        hg.setColorAt(1.0, QColor("#FCFAFF"))
        p.fillRect(QRectF(0, 0, w, 44), hg)
        p.setPen(QPen(QColor("#EFE7FA"), 1.0))
        p.drawLine(QPointF(0, 43.5), QPointF(w, 43.5))
        p.restore()
        # a soft light runs through the header while the popup is open
        p.save()
        p.setClipPath(path)
        cx = -70.0 + (w + 140.0) * (0.5 + 0.5 * math.sin(self._phase * 0.7))
        hg2 = QLinearGradient(cx - 50, 0, cx + 50, 0)
        hg2.setColorAt(0.0, QColor(255, 255, 255, 0))
        hg2.setColorAt(0.5, QColor(255, 255, 255, 30))
        hg2.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillRect(QRectF(0, 0, w, 44), hg2)
        p.restore()
        # live dot and header
        breathe = 0.5 + 0.5 * math.sin(self._phase)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(124, 58, 237, int(28 + 28 * breathe)))
        p.drawEllipse(QPointF(17, 22), 5.5 + 1.0 * breathe, 5.5 + 1.0 * breathe)
        p.setBrush(QColor(124, 58, 237))
        p.drawEllipse(QPointF(17, 22), 2.2, 2.2)
        p.setPen(QColor(C["ink"]))
        p.setFont(_font(13, QFont.Weight.Bold, cn=True))
        p.drawText(QRectF(30, 10, w - 92, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._name or "行业成分")
        pct_color = QColor("#6D28D9" if (self._pct or 0) > 0 else C["muted"])
        p.setPen(pct_color)
        p.setFont(_font(11.5, QFont.Weight.Bold))
        pct_txt = f"{self._pct * 100:+.2f}%" if self._pct is not None else "—"
        p.drawText(QRectF(w - 82, 10, 68, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, pct_txt)
        sub = f"{len(self._stocks)} 只成分 · 按当日涨幅排序" if self._stocks else ("正在加载成分…" if self._loading else "暂无成分缓存")
        p.setPen(QColor(C["muted"]))
        p.setFont(_font(9, QFont.Weight.DemiBold, cn=True))
        p.drawText(QRectF(30, 27, w - 44, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, sub)
        if not self._stocks:
            p.setPen(QColor(C["muted"]))
            p.setFont(_font(10.5, QFont.Weight.Normal, cn=True))
            p.drawText(QRectF(0, 56, w, 28), Qt.AlignmentFlag.AlignCenter,
                       "正在获取实时行情…" if self._loading else "稍后自动重试")
            p.end()
            return
        top = 50.0
        base_opacity = min(1.0, max(0.0, e))
        for i, stock in enumerate(self._stocks[:10]):
            row_t = min(1.0, max(0.0, (base_opacity - i * 0.045) / 0.55))
            if row_t <= 0.0:
                continue
            p.save()
            p.setOpacity(base_opacity * row_t)
            p.translate(0, (1.0 - row_t) * 4.0)
            y = top + i * 21.0
            if i:
                p.setPen(QPen(QColor("#F5F1FA"), 1.0))
                p.drawLine(QPointF(12, y - 4), QPointF(w - 12, y - 4))
            p.setPen(QColor(C["muted"]))
            p.setFont(_font(9.5))
            p.drawText(QRectF(12, y + 1, 16, 17), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, str(i + 1))
            p.setPen(QColor(C["ink2"]))
            p.setFont(_font(11, QFont.Weight.DemiBold, cn=True))
            nm = str(stock.get("name") or stock.get("code") or "—")
            p.drawText(QRectF(32, y + 1, w - 104, 17), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, nm)
            val = stock.get("pct")
            p.setPen(QColor("#6D28D9" if (val or 0) > 0 else C["muted"]))
            p.setFont(_font(10.5, QFont.Weight.Bold))
            p.drawText(QRectF(w - 72, y + 1, 58, 17), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{val * 100:+.2f}%" if val is not None else "—")
            p.restore()
        if len(self._stocks) > 10:
            p.setPen(QColor(C["muted"]))
            p.setFont(_font(9, QFont.Weight.Normal, cn=True))
            p.drawText(QRectF(12, h - 16, w - 24, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       f"+{len(self._stocks) - 10} 只")
        p.end()


class IndustryBoard(Card):
    """Compact SW2 industry leaderboard between candidates and review."""

    def __init__(self, feed: IndustryFeed, popup: IndustryPopup):
        super().__init__()
        self.feed = feed
        self.popup = popup
        self._items: dict[str, dict] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 17, 18, 16)
        lay.setSpacing(7)
        right = QWidget()
        right.setStyleSheet("background:transparent;")
        rl = QHBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(5)
        rl.addWidget(PulseDot(5), 0, Qt.AlignmentFlag.AlignVCenter)
        self.status = _pill("加载中…", C["muted"], "#F8F6FC", C["line"], 9.5, "4px 8px")
        rl.addWidget(self.status, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addLayout(_card_head("S", "二级行业排行", "", right))
        head = QHBoxLayout()
        head.setContentsMargins(5, 0, 6, 0)
        h1 = _label("行业", _font(9, cn=True), C["muted"])
        h2 = _label("当日涨幅", _font(9, cn=True), C["muted"])
        head.addWidget(h1)
        head.addStretch(1)
        head.addWidget(h2)
        lay.addLayout(head)
        self._empty = _label("正在获取申万二级行情…", _font(10.5, cn=True), C["muted"])
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._empty)
        self._rows: list[IndustryRow] = []
        for _ in range(10):
            row = IndustryRow(self._enter_row, self._leave_row)
            self._rows.append(row)
            lay.addWidget(row)
            row.hide()
        lay.addStretch(1)
        self.feed.snapshot_ready.connect(self._apply_snapshot)
        self.feed.members_ready.connect(self._members_ready)
        self.feed.error.connect(self._feed_error)
        if self.feed.cache:
            self._apply_snapshot(self.feed.cache)

    def _apply_snapshot(self, data: dict) -> None:
        items = (data.get("industries") or [])[:10]
        self._items = {str(x.get("code")): x for x in items}
        max_abs = max((abs(float(x.get("pct") or 0)) for x in items), default=0.0)
        for i, row in enumerate(self._rows):
            if i < len(items):
                row.set_data(i + 1, items[i], max_abs)
                row.set_leader(i == 0)
                row.show()
            else:
                row.hide()
        self._empty.setVisible(not bool(items))
        stamp = str(data.get("generated_at") or "")
        hhmm = stamp[11:16] if len(stamp) >= 16 else ""
        if ":" not in hhmm:
            hhmm = ""
        state = str(data.get("trade_state") or "收盘")
        self.status.setText(f"{state} {hhmm}".strip())
        if self.popup.current_code in self._items:
            item = self._items[self.popup.current_code]
            self.popup.update_stocks(self.feed.stocks_for(self.popup.current_code),
                                     self.feed.is_loading(self.popup.current_code))

    def _feed_error(self, message: str) -> None:
        if not self._items:
            self._empty.setText("行业行情暂不可用，稍后自动重试")
            self.status.setText("离线")
        if self.popup.isVisible() and not self.popup._stocks:
            self.popup.update_stocks([], False)

    def _enter_row(self, row: IndustryRow) -> None:
        item = row._item
        code = str(item.get("code") or "")
        if not code:
            return
        self.feed.ensure_members(code)
        self.popup.show_for(
            code,
            _display_industry(str(item.get("name") or "")),
            item.get("pct"),
            self.feed.stocks_for(code),
            row,
            not self.feed.has_members(code) or self.feed.is_loading(code),
        )

    def _leave_row(self, row: IndustryRow) -> None:
        if self.popup.current_code == str(row._item.get("code") or ""):
            self.popup.hide_popup()

    def _members_ready(self, code: str) -> None:
        if self.popup.current_code == code:
            self.popup.update_stocks(self.feed.stocks_for(code), self.feed.is_loading(code))


# ------------------------------------------------------------------ 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TANDEM")
        self.setWindowIcon(QIcon(str(ASSETS / "app.ico")))
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(1080, 700)
        self.resize(1320, 916)
        self._drag_pos: QPoint | None = None
        self._maxed = False
        self._fullscreen = False
        self.state = _read_json(LATEST)
        self._mtime = LATEST.stat().st_mtime if LATEST.exists() else 0.0
        self._proc: QProcess | None = None
        self._running = False
        self._run_started_at = 0.0
        self._finished_shown = False
        self._result_shown = False
        self._error_shown = False
        self._status_tone = "muted"

        self.surface = Surface(self)
        self.setCentralWidget(self.surface)
        self.industry_feed = IndustryFeed(self)
        self.root = QVBoxLayout(self.surface)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(0)

        self.titlebar = TitleBar(self)
        self.root.addWidget(self.titlebar)
        self._sc_f11 = QShortcut(QKeySequence("F11"), self)
        self._sc_f11.activated.connect(self.toggle_fullscreen)
        self._sc_esc = QShortcut(QKeySequence("Esc"), self)
        self._sc_esc.activated.connect(lambda: self.showNormal() if self.isFullScreen() else None)
        self._sc_refresh = QShortcut(QKeySequence("F5"), self)
        self._sc_refresh.activated.connect(self._manual_refresh)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollArea > QWidget > QWidget{background:transparent;}"
            "QScrollBar:vertical{width:8px;background:transparent;margin:4px 2px;}"
            "QScrollBar::handle:vertical{background:#DCD3EA;border-radius:4px;min-height:36px;}"
            "QScrollBar::handle:vertical:hover{background:#C8BBDD;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}"
        )
        self.scroll.viewport().setAutoFillBackground(False)
        self.host = QWidget()
        self.host.setObjectName("Host")
        self.host.setStyleSheet("QWidget#Host{background:transparent;}")
        self.scroll.setWidget(self.host)
        self.content = QVBoxLayout(self.host)
        self.content.setContentsMargins(34, 20, 34, 30)
        self.content.setSpacing(20)
        self.root.addWidget(self.scroll, 1)
        self.industry_popup = IndustryPopup(self.host)
        self.scroll.verticalScrollBar().valueChanged.connect(lambda _=0: self.industry_popup.hide_now())

        self.float_prog = FloatingProgress()
        self.grip = QSizeGrip(self.surface)
        self.grip.setFixedSize(16, 16)
        self.grip.setStyleSheet("background:transparent;")
        self.grip.raise_()

        self.proc_timer = QTimer(self)
        self.proc_timer.timeout.connect(self._tick)
        self.proc_timer.start(350)
        self.file_timer = QTimer(self)
        self.file_timer.timeout.connect(self._check_file)
        self.file_timer.start(900)

        self._render()

    # ---------- window chrome
    def _place_overlays(self):
        # 悬浮球是独立置顶工具窗，不随主窗口移动/调整；这里放尺寸手柄与底部 dock
        self.grip.move(self.width() - 20, self.height() - 20)
        self.grip.setVisible(not self.isFullScreen())

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        if hasattr(self, "industry_popup"):
            self.industry_popup.hide_now()
        self._place_overlays()

    def mousePressEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and not self._maxed:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):  # noqa: N802
        if self._drag_pos is not None and e.buttons() & Qt.MouseButton.LeftButton and not self._maxed:
            if hasattr(self, "industry_popup"):
                self.industry_popup.hide_now()
            self.move(e.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):  # noqa: N802
        self._drag_pos = None
        super().mouseReleaseEvent(e)

    def toggle_max(self):
        if self.isFullScreen():
            self.showNormal()
        elif self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        QTimer.singleShot(30, self._place_overlays)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        QTimer.singleShot(30, self._place_overlays)

    def keyPressEvent(self, e):  # noqa: N802
        if e.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            return
        if e.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):  # noqa: N802
        try:
            self.float_prog.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.industry_feed.shutdown()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(e)

    def changeEvent(self, e):  # noqa: N802
        if e.type() == QEvent.Type.WindowStateChange:
            self._maxed = self.isMaximized()
            self._fullscreen = self.isFullScreen()
            self.surface.maximized = self._maxed or self._fullscreen
            self.surface.update()
            if self.isMinimized():
                _FRAME.pause()
                if hasattr(self, "industry_popup"):
                    self.industry_popup.hide_now()
            else:
                _FRAME.resume()
        elif e.type() == QEvent.Type.ActivationChange and self.isActiveWindow() and not self._running:
            QTimer.singleShot(0, self._check_file)
        super().changeEvent(e)

    def center(self):
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(1320, screen.width() - 40)
        h = min(916, screen.height() - 40)
        self.resize(w, h)
        self.move(screen.x() + (screen.width() - w) // 2, screen.y() + (screen.height() - h) // 2)

    # ---------- data / rendering
    def _set_status(self, text: str, tone: str = "muted"):
        self._status_tone = tone
        self.status_label.setText(text)
        color = {"muted": C["muted"], "ok": C["purple_d"], "warn": C["warn"], "err": "#B42318"}.get(tone, C["muted"])
        self.status_label.setStyleSheet(f"color:{color};background:transparent;")

    def _check_file(self):
        if self._running:
            return
        try:
            m = LATEST.stat().st_mtime
        except OSError:
            return
        if m > self._mtime + 0.01:
            self._mtime = m
            self.state = _read_json(LATEST)
            self._render()
            self._set_status("已同步最新预测 " + str(self.state.get("generated_at", "")), "ok")

    def _render(self):
        try:
            self._render_impl()
        except Exception:  # noqa: BLE001
            self._handle_render_error()

    def _render_impl(self):
        # 关键：递归清理嵌套 layout。旧实现只删 widget，addLayout() 的旧卡片会残留叠影。
        if hasattr(self, "industry_popup"):
            self.industry_popup.hide_now()
        _clear_layout(self.content)
        refs: dict = {}
        self._build_header(refs)
        if not self.state or not self.state.get("short"):
            self._build_empty(refs)
            return
        short = self.state.get("short", {})
        long_ = self.state.get("long", {})
        bar = _section_bar("01", "核心决策", "先看结论，再看依据")
        self.content.addWidget(bar)
        _fade_in(bar, 0)
        self._build_row_core(short, long_, refs)
        bar = _section_bar("02", "判断依据", "理由、模型与回测")
        self.content.addWidget(bar)
        _fade_in(bar, 70)
        self._build_row_logic(short, long_, refs)
        bar = _section_bar("03", "候选与复盘", "候选池与运行状态")
        self.content.addWidget(bar)
        _fade_in(bar, 140)
        self._build_row_bottom(short, refs)
        self.content.addStretch(1)
        if hasattr(self, "industry_popup"):
            self.industry_popup.raise_()

    def _handle_render_error(self):
        tb = traceback.format_exc()
        log = ROOT / "reports" / "runs" / "gui_error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] render failed\n{tb}\n")
        except OSError:
            pass
        if hasattr(self, "status_label"):
            self._set_status("界面渲染失败，详见 reports/runs/gui_error.log", "err")
        # 全自动/预测运行中只记日志，避免弹窗阻塞
        if not self._error_shown and not _auto_mode() and not self._running:
            self._error_shown = True
            last = tb.strip().splitlines()[-1] if tb.strip() else "未知错误"
            QMessageBox.warning(
                self, "TANDEM 界面渲染失败",
                "界面渲染出错，已记录到 reports/runs/gui_error.log\n\n" + last,
            )

    def _copy_button(self, tip: str, text_fn) -> QPushButton:
        btn = IslandButton("复制", "copy")
        btn.setFixedHeight(26)
        btn.setMinimumWidth(46)
        btn.setToolTip(tip)
        btn.clicked.connect(lambda: self._copy_text(text_fn(), btn))
        return btn

    @staticmethod
    def _actions(*widgets) -> QWidget:
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        for w in widgets:
            lay.addWidget(w, 0, Qt.AlignmentFlag.AlignVCenter)
        return box

    def _copy_text(self, text: str, btn: QPushButton):
        QApplication.clipboard().setText(text or "（暂无内容）")
        btn.setText("已复制")
        btn.setEnabled(False)
        QTimer.singleShot(1200, lambda: self._restore_copy(btn))

    @staticmethod
    def _restore_copy(btn: QPushButton):
        try:
            btn.setText("复制")
            btn.setEnabled(True)
        except RuntimeError:
            pass

    def _state_meta(self) -> str:
        st = self.state or {}
        return f"T={st.get('T', '—')}｜生成 {st.get('generated_at', '—')}"

    @staticmethod
    def _pending_reviews() -> int:
        path = ROOT / "reports" / "predictions_ledger.csv"
        if not path.exists():
            return 0
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            return sum(
                1
                for r in rows
                if str(r.get("track", "")).strip() == "short"
                and not str(r.get("actual", "") or "").strip()
            )
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _short_of(short: dict) -> tuple[str, str]:
        code = str(short.get("code", "—"))
        name = str(short.get("name", "") or NAMES.get(code, ""))
        return code, name

    @staticmethod
    def _rise_fall_char(label: str) -> str:
        s = str(label or "")
        if any(k in s for k in ("涨", "偏多", "上升", "多")):
            return "涨"
        if any(k in s for k in ("跌", "偏空", "下降", "空")):
            return "跌"
        return "平"

    def _short_verdict_char(self, short: dict) -> str:
        return self._rise_fall_char(short.get("direction", ""))

    def _long_verdict_char(self, long_: dict) -> str:
        ch = self._rise_fall_char(long_.get("state", ""))
        if ch == "平":
            ch = self._rise_fall_char(long_.get("aux", ""))
        return ch

    def _short_result_text(self, short: dict) -> str:
        code, name = self._short_of(short)
        p_up = float(short.get("win_prob", short.get("p_up", 0.5)) or 0.0)
        exp_ret = float(short.get("exp_ret", 0.0) or 0.0)
        action = str(short.get("action", "—"))
        conf = str(short.get("confidence", "—"))
        direction = str(short.get("direction", "—"))
        risk = short.get("risk", {}) or {}
        evidence = short.get("evidence", {}) or {}
        bt = short.get("backtest", {}) or {}
        entry_label = "买入参考" if action in ("可参与", "轻仓试多") else "观察价"
        close = short.get("close", risk.get("entry", "—"))
        sent = evidence.get("sent")
        lines = [
            "【TANDEM · 短线预测】",
            self._state_meta(),
            f"标的：{code} {name}".rstrip(),
            f"方向：{direction}｜结论：{conf} · {action}",
            f"次日上涨概率：{_fmt_pct(p_up, 1)}｜第一页均值 {_fmt_pct(short.get('p_up_avg'), 1)}",
            f"预期次日涨幅：{_fmt_pct(exp_ret, 2, True)}｜收盘 {close if close is not None else '—'}",
            f"风控参考：{entry_label} {risk.get('entry', '—')}｜止盈 {risk.get('target', '—')}｜止损 {risk.get('stop', '—')}（ATR {risk.get('atr_pct', '—')}%）",
            f"当日涨幅 {_fmt_pp(evidence.get('pct_chg'))}｜5日 {_fmt_pct(evidence.get('ret_5'), 1, True)}｜换手分位 {evidence.get('turn', '—')}｜情绪 {sent if sent is not None else '—'}",
        ]
        if bt:
            gate = f"｜概率≥55%档 {bt.get('gate_days', 0)} 天命中 {_fmt_pct(bt.get('gate_up_rate'), 1)}" if bt.get("gate_days") else ""
            lines.append(f"样本外：近 {bt.get('days', 0)} 日首选上涨命中 {_fmt_pct(bt.get('up_rate'), 1)}｜平均次日 {_fmt_pct(bt.get('avg_ret'), 2, True)}{gate}")
        if short.get("warning"):
            lines.append("纪律提醒：" + str(short["warning"]))
        lines.append("以上为量化研究输出，不构成投资建议。")
        return "\n".join(lines)

    def _long_result_text(self, long_: dict) -> str:
        lines = [
            "【TANDEM · 长线预测｜中国铁物 000927】",
            self._state_meta(),
            f"趋势：{long_.get('state', '—')}｜建议：{long_.get('advice', '—')}",
            f"收盘 {long_.get('close', '—')}｜MA60 {long_.get('ma60', '—')}｜MA200 {long_.get('ma200', '—')}",
            f"52周：距高 {_fmt_pct(long_.get('hi252'), 1, True)}｜距低 {_fmt_pct(long_.get('lo252'), 1, True)}",
            f"明日方向（辅助）：{long_.get('aux', '—')}（集成分 {long_.get('aux_score', '—')}）｜允许涨/跌/中性",
            f"长线框架：{long_.get('pred_note', '长期持有，不因单日涨跌改变框架')}",
        ]
        triggers = long_.get("triggers", []) or []
        if triggers:
            lines.append("触发提醒：" + "；".join(str(x) for x in triggers))
        lines.append("以上为量化研究输出，不构成投资建议。")
        return "\n".join(lines)

    def _short_logic_text(self, short: dict) -> str:
        code, name = self._short_of(short)
        lines = [
            "【TANDEM · 短线预测逻辑】",
            self._state_meta(),
            f"标的：{code} {name}｜次日上涨概率 {_fmt_pct(short.get('win_prob'), 1)}",
            "",
            "预测理由：",
        ]
        for i, r in enumerate(short.get("reasons", []) or [], 1):
            lines.append(f"{i}. {r}")
        if short.get("warning"):
            lines.append("纪律提醒：" + str(short["warning"]))
        lines.append("以上为量化研究输出，不构成投资建议。")
        return "\n".join(lines)

    def _long_logic_text(self, long_: dict) -> str:
        lines = [
            "【TANDEM · 长线预测逻辑｜中国铁物 000927】",
            self._state_meta(),
            "预测理由：",
        ]
        for i, r in enumerate(long_.get("reasons", []) or [], 1):
            lines.append(f"{i}. {r}")
        triggers = long_.get("triggers", []) or []
        if triggers:
            lines.append("触发提醒：" + "；".join(str(x) for x in triggers))
        lines.append("以上为量化研究输出，不构成投资建议。")
        return "\n".join(lines)

    def _build_header(self, refs):
        head = QWidget()
        lay = QHBoxLayout(head)
        lay.setContentsMargins(2, 2, 2, 0)
        lay.setSpacing(16)
        title = ShimmerTitle("今日双轨决策", _font(30, QFont.Weight.Bold, cn=True))
        lay.addWidget(title, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addStretch(1)
        btns = QHBoxLayout()
        btns.setSpacing(9)
        self.refresh_btn = IslandButton("刷新", "ghost")
        self.refresh_btn.setFixedHeight(42)
        self.refresh_btn.setMinimumWidth(78)
        self.refresh_btn.setToolTip("重新读取最新结果（F5）")
        self.refresh_btn.clicked.connect(self._manual_refresh)
        self.run_btn = IslandButton("开始预测", "primary")
        self.run_btn.setFixedHeight(42)
        self.run_btn.setMinimumWidth(128)
        self.run_btn.setToolTip("运行完整管线（约 10–20 分钟）")
        self.run_btn.clicked.connect(self._start_run)
        btns.addWidget(self.refresh_btn)
        btns.addWidget(self.run_btn)
        lay.addLayout(btns)
        generated = str(self.state.get("generated_at", "")) if self.state else ""
        self.status_label = _label(
            ("上次生成 " + generated) if generated else "等待首次预测",
            _font(10.5, cn=True), C["muted"],
        )
        self.status_label.setVisible(False)

        rail = GlowRail()
        rl = QHBoxLayout(rail)
        rl.setContentsMargins(7, 4, 7, 4)
        rl.setSpacing(5)
        self._tool_buttons = {}
        for label, handler, tip_text in (
            ("报告", self._open_report, "打开当前日报"),
            ("台账", self._open_ledger, "打开预测台账 CSV"),
            ("审计", self._open_audit, "打开 Top1 可交易性审计"),
            ("监控", self._open_monitor, "打开运行监控"),
            ("日志", self._open_log, "打开最近运行日志"),
            ("竞猜", self._open_guess, "打开竞猜记录 Excel"),
            ("导出", self._export_png, "导出当前界面 PNG"),
            ("置顶", self._toggle_top, "窗口置顶 / 取消置顶"),
        ):
            b = IslandButton(label, "mini")
            b.setFixedHeight(28)
            b.setMinimumWidth(50)
            b.setToolTip(tip_text)
            b.clicked.connect(handler)
            rl.addWidget(b)
            self._tool_buttons[label] = b
        rl.addStretch(1)
        self._topmost = False
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)
        v.addWidget(head)
        v.addWidget(HeroIsland(self.state or {}, host=self.surface))
        v.addWidget(rail)
        self.content.addWidget(box)

    def _build_empty(self, refs):
        card, lay = _card()
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(12)
        lay.addWidget(IconBadge(64), 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(_label("还没有预测结果", _font(22, QFont.Weight.Bold, cn=True), C["ink"]), 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(_label("点击右上角「开始预测」生成结果。", _font(12.5, cn=True), C["muted"]), 0, Qt.AlignmentFlag.AlignHCenter)
        self.content.addWidget(card)
        _fade_in(card, 50, 320)
        self.content.addStretch(1)

    def _build_row_core(self, short: dict, long_: dict, refs):
        row = QHBoxLayout()
        row.setSpacing(20)

        # ---------- A 短线卡
        a, la = _primary_card()
        direction = str(short.get("direction", "看涨"))
        p_up = float(short.get("win_prob", short.get("p_up", 0.5)) or 0.5)
        exp_ret = float(short.get("exp_ret", 0.0) or 0.0)
        confidence = str(short.get("confidence", "—"))
        action = str(short.get("action", "—"))
        risk = short.get("risk", {}) or {}
        code = str(short.get("code", "—"))
        name = str(short.get("name", "") or NAMES.get(code, ""))
        close_pick = float(short.get("close", risk.get("entry", 0.0)) or 0.0)
        trade_p = float((short.get("backtest", {}) or {}).get("trade_p", 0.55) or 0.55)
        dir_bg = C["purple"] if direction == "看涨" else (C["down_bg"] if direction == "看跌" else "#F7F4FC")
        dir_fg = "#FFFFFF" if direction == "看涨" else C["down"]
        la.addLayout(_card_head("A", "短线结论", "", None))

        hero = QHBoxLayout()
        hero.setSpacing(12)
        left = QVBoxLayout()
        left.setSpacing(6)
        code_row = QHBoxLayout()
        code_row.setSpacing(9)
        code_lab = _label(code, _font(36, QFont.Weight.Bold), C["ink"])
        code_lab.setStyleSheet("color:#16121F;background:transparent;letter-spacing:0.8px;")
        code_row.addWidget(code_lab)
        if name:
            nm = _label(name, _font(20, QFont.Weight.DemiBold, cn=True), C["ink"])
            nm.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            code_row.addWidget(nm, 0, Qt.AlignmentFlag.AlignVCenter)
            code_row.addWidget(self._copy_button("复制公司名称", lambda: name or code), 0, Qt.AlignmentFlag.AlignVCenter)
        code_row.addStretch(1)
        left.addLayout(code_row)
        tag_row = QHBoxLayout()
        tag_row.setSpacing(7)
        tag_row.addWidget(_pill(direction, dir_fg, dir_bg, None if direction == "看涨" else C["line"], 12, "5px 13px"))
        tag_row.addWidget(self._copy_button("复制短线结果（一个字：涨/跌/平）", lambda: self._short_verdict_char(short)))
        tag_row.addWidget(_pill(f"{confidence} · {action}", C["purple_d"], C["lav"], C["lav2"], 11, "4px 10px"))
        tag_row.addStretch(1)
        left.addLayout(tag_row)
        hero.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(0)
        right.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        cap = _label("次日上涨概率", _font(11, cn=True), C["muted"])
        cap.setAlignment(Qt.AlignmentFlag.AlignRight)
        right.addWidget(cap)
        big = CountUpLabel(p_up, 1, _font(44, QFont.Weight.Bold), "#6D28D9")
        big.setAlignment(Qt.AlignmentFlag.AlignRight)
        big.setStyleSheet("color:#6D28D9;background:transparent;letter-spacing:-1px;")
        right.addWidget(big)
        hero.addLayout(right)
        la.addLayout(hero)
        la.addWidget(ProbGauge(p_up, trade_p))

        tiles = QHBoxLayout()
        tiles.setSpacing(9)
        tiles.addWidget(StatTile("收盘价", f"{close_pick:.2f}" if close_pick else "—", "", False))
        tiles.addWidget(StatTile("预期次日", _fmt_pct(exp_ret, 2, True), "", exp_ret > 0))
        tiles.addWidget(StatTile("模型结论", confidence, "", p_up >= trade_p))
        tiles.addWidget(StatTile("日波动 ATR", f"{float(risk.get('atr_pct', 0) or 0):.1f}%", "", False))
        la.addLayout(tiles)
        row.addWidget(a, 155)
        _fade_in(a, 30, 300)

        # ---------- B 长线卡
        b, lb = _primary_card()
        state = str(long_.get("state", "震荡"))
        advice = str(long_.get("advice", "长线持有"))
        st_fg = {"上升": C["purple_d"], "下降": C["down"], "震荡": C["warn"]}.get(state, C["purple_d"])
        lb.addLayout(_card_head("B", "长线结论 · 中国铁物", "", None, "#FFF7EA", "#B45309"))
        lh = QHBoxLayout()
        lvv = QVBoxLayout()
        lvv.setSpacing(4)
        state_row = QHBoxLayout()
        state_row.setSpacing(8)
        state_row.addWidget(_label(state + "趋势", _font(25, QFont.Weight.Bold, cn=True), st_fg))
        state_row.addWidget(self._copy_button("复制长线结果（一个字：涨/跌/平）", lambda: self._long_verdict_char(long_)))
        state_row.addStretch(1)
        lvv.addLayout(state_row)
        lvv.addWidget(_label(advice, _font(11.5, cn=True), C["ink2"], wrap=True))
        lh.addLayout(lvv, 1)
        close = float(long_.get("close", 0) or 0)
        cbig = _label(f"{close:.2f}", _font(28, QFont.Weight.Bold), C["ink"])
        cbig.setAlignment(Qt.AlignmentFlag.AlignRight)
        lh.addWidget(cbig, 0, Qt.AlignmentFlag.AlignTop)
        lb.addLayout(lh)
        series = long_.get("series", []) or []
        spark_vals = [float(x.get("c", 0)) for x in series if x.get("c") is not None]
        ma_vals = [x.get("ma60") for x in series] if series and any(x.get("ma60") is not None for x in series) else None
        lb.addWidget(MiniSparkline(spark_vals, ma_vals, 108))
        row.addWidget(b, 100)
        _fade_in(b, 100, 340)
        self.content.addLayout(row)

    def _build_row_logic(self, short: dict, long_: dict, refs):
        row = QHBoxLayout()
        row.setSpacing(20)
        a, la = _card()
        la.addLayout(_card_head("A", "短线为什么选它", "", self._actions(
            self._copy_button("复制短线预测逻辑", lambda: self._short_logic_text(short)))))
        reasons = short.get("reasons", []) or ["等待完整预测生成逻辑小结"]
        la.addWidget(ReasonList(reasons[:7]))
        row.addWidget(a, 155)
        _fade_in(a, 70, 310)

        b, lb = _card()
        lb.addLayout(_card_head("B", "长线持有逻辑", "", self._actions(
            self._copy_button("复制长线预测逻辑", lambda: self._long_logic_text(long_))), "#FFF7EA", "#B45309"))
        long_reasons = long_.get("reasons", []) or ["等待完整预测生成长线小结"]
        lb.addWidget(ReasonList(long_reasons[:7], compact=True))
        row.addWidget(b, 100)
        _fade_in(b, 150, 350)
        self.content.addLayout(row)

    def _build_row_bottom(self, short: dict, refs):
        row = QHBoxLayout()
        row.setSpacing(20)

        # 候选池
        a, la = _card()
        la.addLayout(_card_head("C", "第一页候选池", "", None))
        fp = short.get("first_page", []) or []
        table = QTableWidget(len(fp), 5)
        table.setHorizontalHeaderLabels(["#", "标的", "当日涨幅", "上涨概率", "预期次日"])
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        table.setMouseTracking(True)
        table.viewport().setMouseTracking(True)
        table.setFixedHeight(34 + max(1, len(fp)) * 32 + 4)
        table._row_highlight = TableRowHighlight(table)
        table._hover_filter = TableHoverFilter(table, table._row_highlight)
        table.viewport().installEventFilter(table._hover_filter)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for i in (2, 3, 4):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(0, 32)
        table.setColumnWidth(2, 76)
        table.setColumnWidth(3, 76)
        table.setColumnWidth(4, 76)
        table.setStyleSheet(
            "QTableWidget{background:transparent;border:none;outline:none;}"
            "QTableWidget::item{background:transparent;border-bottom:1px solid #F3EFF9;padding:3px 5px;color:#2B2635;}"
            # 表头横幅：深紫渐变（原先 transparent 会随系统深色调色板变黑）
            "QHeaderView{background:#4C1D95;border:none;}"
            "QHeaderView::section{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #5B21B6,stop:1 #4C1D95);"
            "border:none;border-right:1px solid #6D28D9;border-bottom:1px solid #3B0F7A;"
            "color:#F5F3FF;padding:8px 5px;font-size:10px;font-weight:600;}"
        )
        header.setFixedHeight(36)
        header.setFont(_font(10, QFont.Weight.DemiBold, cn=True))
        table._header_sheen = HeaderSheen(header)
        for r, item in enumerate(fp):
            row_name = item.get("name") or NAMES.get(str(item.get("code", "")), "")
            cells = [
                str(r + 1),
                f"{item.get('code', '')}{(' · ' + row_name) if row_name else ''}",
                _fmt_pp(item.get("pct")),
                _fmt_pct(item.get("p_up"), 1),
                _fmt_pct(item.get("exp_ret"), 2, True),
            ]
            for cidx, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter if cidx != 1 else Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                cell.setFont(_font(11, QFont.Weight.DemiBold if cidx in (1, 3) else QFont.Weight.Normal, cn=False))
                if cidx == 3:
                    cell.setForeground(QColor(C["purple_d"]))
                elif cidx == 2:
                    try:
                        cell.setForeground(QColor(C["purple_d"] if float(item.get("pct", 0)) > 0 else C["down"]))
                    except Exception:  # noqa: BLE001
                        pass
                if r == 0:
                    cell.setBackground(QColor("#F6F1FE"))
                table.setItem(r, cidx, cell)
            table.setRowHeight(r, 31)
        if not fp:
            table.setRowCount(1)
        else:
            table._aura = TableAura(table)
        la.addWidget(table)
        row.addWidget(a, 138)
        _fade_in(a, 120, 320)

        board = IndustryBoard(self.industry_feed, self.industry_popup)
        row.addWidget(board, 98)
        _fade_in(board, 170, 320)

        # 复盘 & 运行
        b, lb = _card()
        review = self.state.get("review", {}) or {}
        lb.addLayout(_card_head("R", "复盘与运行", "短线只认上涨；长线辅助单独统计", None, "#F0E9FF", "#6D28D9"))
        total = int(review.get("total", 0) or 0)
        hits = int(review.get("hits", 0) or 0)
        acc = hits / total if total else 0.0
        score_row = QHBoxLayout()
        if total:
            score = CountUpLabel(hits, 0, _font(30, QFont.Weight.Bold), C["purple_d"], suffix="", mult=1.0)
            score_row.addWidget(score)
            score_row.addWidget(_label(f"/{total}", _font(30, QFont.Weight.Bold), C["ink2"]))
        else:
            score_row.addWidget(_label("待验证", _font(30, QFont.Weight.Bold), C["ink2"]))
        score_row.addStretch(1)
        score_row.addWidget(RateOrb(acc, hits, total), 0, Qt.AlignmentFlag.AlignVCenter)
        lb.addLayout(score_row)
        lb.addWidget(_label(str(review.get("text", "首日运行，等待次日复盘")), _font(11.5, cn=True), C["ink2"], wrap=True))
        pending = self._pending_reviews()
        if pending:
            pending_row = QHBoxLayout()
            pending_row.setSpacing(6)
            pending_row.addWidget(PulseDot(5), 0, Qt.AlignmentFlag.AlignVCenter)
            pending_row.addWidget(_pill(f"待复盘 {pending} 条短线", C["purple_d"], C["lav"], C["lav2"], 10.5, "5px 10px"), 0, Qt.AlignmentFlag.AlignVCenter)
            pending_row.addStretch(1)
            lb.addLayout(pending_row)

        row.addWidget(b, 74)
        _fade_in(b, 210, 340)
        self.content.addLayout(row)

    def _build_guess_card(self, refs):
        """竞猜双轨卡：自选股玩法（长线铁物 + 自选每日涨/跌，猜对+50）。读 latest_guess.json，缺失时静默不渲染。"""
        g = _read_json(LATEST_GUESS)
        if not g or not g.get("rows"):
            return
        card, la = _card()
        st = g.get("stats", {})
        acc = st.get("acc")
        acc_txt = f"{acc:.0%}" if isinstance(acc, (int, float)) else "—"
        la.addLayout(_card_head("G", "竞猜双轨 · 长线 + 自选", "与桌面《股票预测》手写表同款玩法 · 猜对 +50 分 · 结果次日自动核验",
                                _pill(f"累计 {st.get('hits', 0)}✔{('+' + str(st.get('halves', 0)) + '½') if st.get('halves') else ''}/{st.get('total', 0)} = {st.get('reward', 0)} 分 · 命中 {acc_txt}",
                                      C["ink2"], "#F8F6FC", C["line"], 10, "4px 9px")))
        for r in g.get("rows", []):
            track = r.get("轨道", "")
            pred = r.get("明日预测", "")
            pu = r.get("p_up")
            pu_txt = f"{pu * 100:.1f}%" if isinstance(pu, (int, float)) else "—"
            fg = "#6D28D9" if pred == "涨" else ("#8A8794" if pred == "平" else "#5B5560")
            bg = "#F3EEFF" if pred == "涨" else ("#F6F4F9" if pred == "平" else "#F1EFF4")
            row = QHBoxLayout()
            row.setSpacing(10)
            tag = _pill(track or "自选", "#6D28D9" if track == "长线" else "#7A7486", "#F3EEFF" if track == "长线" else "#F5F3F8", None, 10.5, "3px 9px")
            row.addWidget(tag)
            row.addWidget(_label(str(r.get("股票", "")), _font(12.5, QFont.Weight.DemiBold, cn=True), C["ink"]))
            note = str(r.get("note", "") or "")
            if note:
                row.addWidget(_label(note, _font(9.5, cn=True), C["muted"]))
            row.addStretch(1)
            row.addWidget(_label(f"p_up {pu_txt}", _font(11, cn=True), C["muted"]))
            row.addWidget(_pill(("▲ " if pred == "涨" else ("● " if pred == "平" else "▼ ")) + pred, fg, bg, None, 11.5, "5px 13px"))
            la.addLayout(row)
        wk = [x for x in (g.get("weekly") or []) if x.get("股票") == "中国铁物"]
        if wk:
            x = wk[0]
            la.addWidget(_label(f"周小结：{x.get('股票')} 本周 {x.get('本周涨跌')} · {x.get('趋势')} · 下周 {x.get('下周预测')}",
                                _font(10, cn=True), C["muted"], wrap=True))
        self.content.addWidget(card)
        _fade_in(card, 210)

    def _build_footer(self, refs):
        fw = QWidget()
        lay = QVBoxLayout(fw)
        lay.setContentsMargins(4, 2, 4, 0)
        lay.setSpacing(4)
        line = _label(
            "风险提示：本软件为个人量化研究工具，所有概率与收益均为模型估计，不构成投资建议。短线概率未达出手线时，纪律优先于交易冲动；中国铁物按长期持有管理，单日方向不代表长期价值。",
            _font(9.5, cn=True), C["muted"], wrap=True,
        )
        line.setStyleSheet("color:#9A93A5;background:transparent;line-height:145%;")
        lay.addWidget(line)
        meta = _label("TANDEM · 本地原生桌面版 · 每日 17:30 自动快照 / 04:30 自学习", _font(9.5, cn=True), C["muted"])
        meta.setStyleSheet("color:#B2ABBD;background:transparent;")
        lay.addWidget(meta)
        self.content.addWidget(fw)

    # ---------- actions
    def _open_local(self, rel: str, label: str):
        path = ROOT / rel
        if not path.exists():
            self._set_status(f"{label}不存在：{rel}", "err")
            return
        if _auto_mode():
            self._set_status(f"{label}已定位（自动模式不打开外部窗口）", "ok")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self._set_status(f"已打开{label}", "ok")

    def _open_report(self):
        t = str((self.state or {}).get("T", ""))
        candidate = ROOT / "reports" / f"{t}.md" if t else None
        if candidate is not None and candidate.exists():
            self._open_local(str(candidate.relative_to(ROOT)), "日报")
        else:
            self._open_local("reports/top1_audit/report.md", "审计报告")

    def _open_ledger(self):
        self._open_local("reports/predictions_ledger.csv", "台账")

    def _open_audit(self):
        self._open_local("reports/top1_audit/report.md", "审计")

    def _open_monitor(self):
        self._open_local("reports/monitor/latest.md", "监控")

    def _open_log(self):
        self._open_local("reports/runs/last_run.log", "日志")

    def _open_guess(self):
        self._open_local("reports/竞猜记录.xlsx", "竞猜记录")

    def _export_png(self):
        t = str((self.state or {}).get("T", "") or time.strftime("%Y-%m-%d"))
        dest = ROOT / "reports" / "runs" / f"gui_export_{t}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok = self.grab().save(str(dest))
        self._set_status(f"已导出 {dest.relative_to(ROOT)}" if ok else "导出失败", "ok" if ok else "err")

    def _toggle_top(self):
        self._topmost = not bool(getattr(self, "_topmost", False))
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, self._topmost)
        self.show()
        btn = getattr(self, "_tool_buttons", {}).get("置顶")
        if btn is not None:
            btn.setText("取消置顶" if self._topmost else "置顶")
        self._set_status("窗口已置顶" if self._topmost else "已取消置顶", "ok")

    def _manual_refresh(self):
        self.state = _read_json(LATEST)
        try:
            self._mtime = LATEST.stat().st_mtime
        except OSError:
            self._mtime = 0
        self._render()
        self._set_status("已刷新 " + str(self.state.get("generated_at", "")), "ok")

    def _start_run(self):
        if self._running:
            return
        self._running = True
        self._run_started_at = time.perf_counter()
        self._finished_shown = False
        self._result_shown = False
        self.run_btn.setEnabled(False)
        self.run_btn.setText("预测进行中…")
        self.refresh_btn.setEnabled(False)
        self._set_status("完整管线运行中，预计 10–20 分钟…", "warn")
        self.float_prog.begin(self.screen().availableGeometry() if self.screen() else None)
        PROG.parent.mkdir(parents=True, exist_ok=True)
        try:
            PROG.write_text(json.dumps({"pct": 0.5, "stage": "启动", "msg": "准备完整管线", "ts": time.strftime("%H:%M:%S")}, ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError:
            pass
        py = ROOT / ".venv" / "Scripts" / "pythonw.exe"
        if not py.exists():
            py = Path(sys.executable)
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(ROOT))
        self._proc.setProgram(str(py))
        self._proc.setArguments(["-m", "src.report.run_full"])
        env = QProcessEnvironment.systemEnvironment()
        env.insert("OMP_NUM_THREADS", "4")
        env.insert("MKL_NUM_THREADS", "4")
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")  # gui_run.log 统一 UTF-8，避免 GBK 混合
        self._proc.setProcessEnvironment(env)
        RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        self._proc.setStandardOutputFile(str(RUN_LOG), QIODevice.OpenModeFlag.Append)
        self._proc.setStandardErrorFile(str(RUN_LOG), QIODevice.OpenModeFlag.Append)
        self._proc.finished.connect(self._on_proc_finished)
        self._proc.errorOccurred.connect(self._on_proc_error)
        self._proc.start()

    def _tick(self):
        if not self._running:
            return
        d = _read_progress()
        self.float_prog.update_progress(d)
        if d.get("pct", 0) >= 100 and not self._finished_shown:
            # 100% 是管线最后一个事件，latest.json 此时已落盘；
            # 不再等待进程退出（CUDA 清理可能还要数秒），立即读取并渲染。
            self._finished_shown = True
            self.float_prog.finish()
            self._after_finish()
        if (
            not self._finished_shown
            and (self._proc is None or self._proc.state() == QProcess.ProcessState.NotRunning)
            and time.perf_counter() - self._run_started_at > 4.0
        ):
            self._recover_stuck_run()

    def _recover_stuck_run(self):
        """子进程已消失但 finished 信号丢失时，避免界面永久停在运行中。"""
        if not self._running:
            return
        self._running = False
        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始预测")
        self.refresh_btn.setEnabled(True)
        self.float_prog.hide()
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.deleteLater()
            except Exception:  # noqa: BLE001
                pass
        self._set_status("检测到预测进程已结束，界面已恢复；如无新结果请重试", "warn")
        QTimer.singleShot(0, self._manual_refresh)

    def _on_proc_finished(self, code: int, status):  # noqa: N802
        if not self._running:
            return
        self._running = False
        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始预测")
        self.refresh_btn.setEnabled(True)
        if code == 0:
            self.float_prog.finish()
            QTimer.singleShot(0, self._after_finish)
        else:
            self.float_prog.hide()
            self._set_status(f"运行未完成（退出码 {code}），详见 reports/runs/gui_run.log", "err")
            QTimer.singleShot(500, self._manual_refresh)

    def _on_proc_error(self, err):
        if not self._running:
            return
        self._running = False
        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始预测")
        self.refresh_btn.setEnabled(True)
        self.float_prog.hide()
        self._set_status(f"启动失败：{err}", "err")

    def _apply_latest(self):
        try:
            self._mtime = LATEST.stat().st_mtime
        except OSError:
            self._mtime = 0
        self.state = _read_json(LATEST)
        self._render()

    def _smooth_top(self):
        sb = self.scroll.verticalScrollBar()
        if sb.value() == 0:
            return
        anim = QPropertyAnimation(sb, b"value", self)
        anim.setDuration(420)
        anim.setStartValue(sb.value())
        anim.setEndValue(0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._scroll_anim = anim
        anim.start()

    def _after_finish(self):
        if self._result_shown:
            return
        self._result_shown = True
        self._apply_latest()
        self._smooth_top()
        self._set_status("预测完成 · 结果已刷新 " + str(self.state.get("generated_at", "")), "ok")


class Surface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.maximized = False

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        radius = 0 if self.maximized else 22
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), radius, radius)
        p.setClipPath(path)
        bg = QLinearGradient(0, 0, w, h)
        bg.setColorAt(0, QColor("#FCFBFF"))
        bg.setColorAt(0.55, QColor("#FAF8FE"))
        bg.setColorAt(1, QColor("#F4EFFB"))
        p.fillPath(path, bg)
        glow = QRadialGradient(QPointF(w * 0.82, -40), max(w, 600) * 0.55)
        glow.setColorAt(0, QColor(196, 181, 253, 80))
        glow.setColorAt(0.55, QColor(233, 221, 255, 30))
        glow.setColorAt(1, QColor(255, 255, 255, 0))
        p.fillPath(path, glow)
        p.setClipping(False)
        p.setPen(QPen(QColor("#E7E0F1"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        if not self.maximized:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 22))
            p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), radius, radius)
        p.end()


class TitleBar(QWidget):
    def __init__(self, win: MainWindow):
        super().__init__()
        self.win = win
        self.setFixedHeight(62)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 14, 0)
        lay.setSpacing(10)
        logo = IconBadge(32)
        lay.addWidget(logo)
        t = _label("TANDEM", _font(15.5, QFont.Weight.Bold), C["ink"])
        lay.addWidget(t)
        lay.addStretch(1)

        self.min_btn = IslandButton("—", "win")
        self.min_btn.setFixedSize(34, 30)
        self.min_btn.setToolTip("最小化")
        self.max_btn = IslandButton("□", "win")
        self.max_btn.setFixedSize(34, 30)
        self.max_btn.setToolTip("最大化 / 还原（F11 全屏）")
        self.close_btn = IslandButton("×", "close")
        self.close_btn.setFixedSize(34, 30)
        self.close_btn.setToolTip("关闭")
        self.min_btn.clicked.connect(win.showMinimized)
        self.max_btn.clicked.connect(win.toggle_max)
        self.close_btn.clicked.connect(win.close)
        for b in (self.min_btn, self.max_btn, self.close_btn):
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignVCenter)
        self._phase = 0.0
        _FRAME.subscribe(self._beat)

    def _beat(self, frame: int):
        if not self.isVisible():
            return
        self._phase = (self._phase + 0.024) % (math.pi * 2)
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        y = self.height() - 1
        p.setPen(QPen(QColor("#EDE7F5"), 1))
        p.drawLine(0, y, self.width(), y)
        cx = self.width() * (0.5 + 0.5 * math.sin(self._phase * 0.7))
        g = QLinearGradient(cx - 80, 0, cx + 80, 0)
        g.setColorAt(0.0, QColor(167, 139, 250, 0))
        g.setColorAt(0.5, QColor(124, 58, 237, 105))
        g.setColorAt(1.0, QColor(167, 139, 250, 0))
        p.setPen(QPen(QBrush(g), 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(0, y), QPointF(self.width(), y))
        p.end()

    def mousePressEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton and not self.win._maxed and not self.win.isFullScreen():
            self.win._drag_pos = e.globalPosition().toPoint() - self.win.frameGeometry().topLeft()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):  # noqa: N802
        if self.win._drag_pos is not None and e.buttons() & Qt.MouseButton.LeftButton and not self.win._maxed and not self.win.isFullScreen():
            if hasattr(self.win, "industry_popup"):
                self.win.industry_popup.hide_now()
            self.win.move(e.globalPosition().toPoint() - self.win._drag_pos)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):  # noqa: N802
        self.win._drag_pos = None
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.win.toggle_max()
        super().mouseDoubleClickEvent(e)


APP_QSS = f"""
* {{ outline: none; }}
QWidget {{ font-family:"{FONT_CN}","Segoe UI"; color:{C['ink']}; }}
QLabel {{ background:transparent; }}
QFrame#Card {{
    background:{C['card']};
    border:1px solid {C['line']};
    border-radius:20px;
}}
QFrame#Tile {{ background:#FCFBFE; border:1px solid #F1ECF8; border-radius:13px; }}
QFrame#ReasonRow {{ background:transparent; border:none; border-bottom:1px solid #F3EFF9; border-radius:0; }}
QFrame#PlanBox {{ background:#FCFBFE; border:1px solid #F0EAF9; border-radius:13px; }}
QFrame#SoftBox {{ background:#F7F3FF; border:1px dashed #E3D8F7; border-radius:13px; }}
QFrame#WarnBox {{ background:{C['warn_bg']}; border:1px solid {C['warn_line']}; border-radius:12px; }}
QPushButton#Primary {{
    color:#FFFFFF;
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #8B5CF6,stop:1 #6D28D9);
    border:none; border-radius:12px; padding:0 20px;
    font-family:"{FONT_CN}"; font-size:12.5px; font-weight:600;
}}
QPushButton#Primary:hover {{
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7C3AED,stop:1 #5B21B6);
}}
QPushButton#Primary:disabled {{ background:#D8CCF2; color:#FFFFFF; }}
QPushButton#Ghost {{
    color:{C['ink2']}; background:#FFFFFF; border:1px solid {C['line']};
    border-radius:12px; padding:0 16px;
    font-family:"{FONT_CN}"; font-size:12px;
}}
QPushButton#Ghost:hover {{ background:#FAF7FF; border-color:{C['lav2']}; }}
QToolButton#WinBtn {{
    color:#8D869A; background:transparent; border:none; border-radius:8px;
    font-size:14px; font-weight:600;
}}
QToolButton#WinBtn:hover {{ background:#F2EDFA; color:{C['purple_d']}; }}
QToolButton#WinClose {{
    color:#8D869A; background:transparent; border:none; border-radius:8px;
    font-size:17px; font-weight:400;
}}
QToolButton#WinClose:hover {{ background:#FDECEC; color:#C53030; }}
QPushButton#Copy {{
    color:{C['purple_d']}; background:#F7F2FF; border:1px solid #E7DDF7;
    border-radius:8px; padding:0 9px;
    font-family:"{FONT_CN}"; font-size:10.5px; font-weight:600;
}}
QPushButton#Copy:hover {{ background:#EFE7FF; border-color:{C['lav2']}; }}
QPushButton#Copy:disabled {{ color:#9A93A5; background:#F5F2FA; border-color:#EDE8F5; }}
"""


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    screenshot = None
    size = None
    if "--screenshot" in argv:
        i = argv.index("--screenshot")
        if i + 1 < len(argv):
            screenshot = argv[i + 1]
    if "--size" in argv:
        i = argv.index("--size")
        if i + 1 < len(argv):
            size = argv[i + 1]

    try:
        if sys.platform == "win32":
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DailyStockOracle.Desktop")
    except Exception:  # noqa: BLE001
        pass
    app = QApplication([sys.argv[0]])
    app.setApplicationName("TANDEM")
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(ASSETS / "app.ico")))
    app.setStyleSheet(APP_QSS)

    def _hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log = ROOT / "reports" / "runs" / "gui_error.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] unhandled\n{text}\n")
        except OSError:
            pass
        if _auto_mode():
            return
        try:
            QMessageBox.critical(None, "TANDEM 运行错误",
                                 "程序出现未处理错误，已记录到 reports/runs/gui_error.log\n\n" + str(exc))
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = _hook
    win = MainWindow()
    if size:
        try:
            w, h = (int(x) for x in size.lower().split("x"))
            win.resize(w, h)
        except Exception:  # noqa: BLE001
            win.center()
    else:
        win.center()
    win.show()

    if screenshot:
        def grab():
            Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
            win.grab().save(screenshot)
            app.quit()

        QTimer.singleShot(1400, grab)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
