"""特征基础设施：时间戳审计装饰器（docs/03 §3.2 防泄漏工程规范）。

每个特征函数声明 available_at（信息可得时点）。审计原则：
任何特征在 T 日的取值，仅能由 date<=T 的行计算得出。
tests/test_leakage.py 通过"截断数据重算 vs 全量数据取行"一致性校验强制执行。
"""
from collections.abc import Callable

REGISTRY: dict[str, dict] = {}


def feature_info(available_at: str = "T_close") -> Callable:
    """装饰器：登记特征的合法信息时点。available_at 只允许 'T_close'（P2 全部特征）。"""

    def deco(fn: Callable) -> Callable:
        REGISTRY[fn.__name__] = {"available_at": available_at}
        return fn

    return deco
