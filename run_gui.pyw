import ctypes
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
try:
    from src.gui.desktop import main
    main()
except Exception:
    log = os.path.join(ROOT, "reports", "runs", "gui_error.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n" + traceback.format_exc())
    try:
        ctypes.windll.user32.MessageBoxW(0, "界面启动失败，详情见：\n" + log, "双轨智测", 0x10)
    except Exception:  # noqa: BLE001
        pass
    raise
