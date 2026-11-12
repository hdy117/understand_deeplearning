"""日志工具：基于 loguru,一次配置,控制台 + 文件双写。

用法
----
>>> from utils import setup_logger
>>> setup_logger(log_dir="logs", level="DEBUG")
>>> from loguru import logger
>>> logger.info("训练开始")

或者在模块里只拿 logger（首次调用自动用默认配置）
>>> from utils import get_logger
>>> logger = get_logger()
>>> with log_time(logger, "第 1 个 epoch"):
...     ...
"""

from __future__ import annotations

import sys
import time
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import loguru
import torch
from loguru import logger

__all__ = [
    "logger",
    "setup_logger",
    "get_logger",
    "Timer",
    "log_time",
    "set_seed",
    "LOG_FILE",
    "DEFAULT_LOG_DIR",
]

#: 日志目录
DEFAULT_LOG_DIR = "logs"

#: 控制台格式：时间 | 级别 | 模块:函数:行号 | 内容
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
#: 文件格式：纯文本,方便 grep
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"

#: 当前日志文件路径（未启用文件 sink 时为 None）
LOG_FILE: Path | None = None

_configured = False


def setup_logger(
    log_dir: str | Path | None = DEFAULT_LOG_DIR,
    level: str | int = "INFO",
    *,
    name: str = "flow_matching",
    filename: str | None = None,
    console: bool = True,
    stream: TextIO = sys.stderr,
    rotation: str | int = "10 MB",
    retention: str | int = "10 days",
    compression: str = "zip",
    enqueue: bool = True,
    colorize: bool | None = None,
) -> "logger.__class__":
    """装配日志 sink（输出目的地）：控制台一个、文件一个。

    ``loguru`` 自带一个 stderr sink,先 ``remove()`` 再重装,所以重复调用
    不会出现日志翻倍,最后一次调用生效（等价于幂等 + force）。

    参数
    ----
    log_dir:   日志目录,``None`` 表示只输出到控制台。
    level:     最低级别,``"DEBUG"`` / ``"INFO"`` / ``"WARNING"`` ...
    filename:  文件名,缺省 ``<name>_{time:YYYYMMDD-HHmmss}.log``;
               保留 ``{time:...}`` 占位符则每次运行自动新建文件。
    rotation:  单个文件的条件,``"10 MB"`` / ``"1 day"`` / ``"00:00"``。
    retention: 旧文件保留策略,``"10 days"`` / ``"1 month"`` / ``5``。
    compression: 旧文件压缩格式,``"zip"`` / ``"gz"`` / ``None``。
    enqueue:   多进程/多线程安全写文件（训练常用 DataLoader worker）。
    colorize:  ``None`` 自动判断是否 TTY。

    返回
    ----
    loguru 的 ``logger``,可直接 ``logger.info(...)``。
    """
    global LOG_FILE, _configured

    logger.remove()  # 去掉 loguru 默认 sink,避免重复输出

    if console:
        logger.add(
            stream,
            level=level,
            format=_CONSOLE_FORMAT,
            colorize=colorize,
            backtrace=True,
            diagnose=True,
        )

    LOG_FILE = None
    if log_dir is not None:
        if filename is None:
            filename = f"{name}_{{time:YYYYMMDD-HHmmss}}.log"
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        LOG_FILE = path / filename
        logger.add(
            str(LOG_FILE),
            level=level,
            format=_FILE_FORMAT,
            rotation=rotation,
            retention=retention,
            compression=compression,
            encoding="utf-8",
            enqueue=enqueue,
            backtrace=True,
            diagnose=True,
        )

    _configured = True
    return logger


def get_logger() -> loguru.Logger:
    """取 logger;若尚未配置,先用默认参数初始化一次。"""
    if not _configured:
        setup_logger()
    return logger


def set_seed(seed: int) -> None:
    """固定 Python 与 PyTorch 的随机种子,让训练尽量可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Timer:
    """用 ``time.perf_counter()`` 测量一段代码的墙上时间。

    既可以手动开始和停止,也可以作为上下文管理器使用::

        timer = Timer()
        timer.start()
        ...
        elapsed = timer.stop()

        with Timer() as timer:
            ...
        print(timer.elapsed)

    计时单位是秒。``elapsed`` 在停止前为 ``None``。同一个实例可以重复使用;
    新的 ``start()`` 会清空上一次的结果。重复启动、未启动就停止或重复停止时
    会抛出 ``RuntimeError``,避免静默返回错误读数。
    """

    def __init__(self) -> None:
        self._started_at: float | None = None
        self._elapsed: float | None = None

    @property
    def elapsed(self) -> float | None:
        """最近一次完整计时的秒数;尚未停止时为 ``None``。"""
        return self._elapsed

    def start(self) -> "Timer":
        """开始一次新计时,并返回自身以便链式调用。"""
        if self._started_at is not None:
            raise RuntimeError("计时已经开始,请先调用 stop()")
        self._started_at = time.perf_counter()
        self._elapsed = None
        return self

    def stop(self) -> float:
        """停止计时并返回本次耗时,单位为秒。"""
        if self._started_at is None:
            raise RuntimeError("计时尚未开始")
        self._elapsed = time.perf_counter() - self._started_at
        self._started_at = None
        return self._elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()


@contextmanager
def log_time(action: str, level: str = "INFO") -> Iterator[None]:
    """记录一段代码的耗时;抛异常时记 ``exception`` 并原样抛出。

    >>> with log_time("训练 1 个 epoch"):
    ...     ...
    2026-11-11 15:02:01.123 | INFO | utils:log_time:78 | 完成：训练 1 个 epoch,耗时 1.234 s
    """
    start = time.perf_counter()
    logger.debug("开始：{}", action)
    try:
        yield
    except BaseException:
        logger.exception(
            "失败：{}（耗时 {:.3f} s）", action, time.perf_counter() - start
        )
        raise
    else:
        logger.log(
            level, "完成：{}（耗时 {:.3f} s）", action, time.perf_counter() - start
        )


# ---------------------------------------------------------------------------
# 演示：python utils.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logger(
        log_dir=DEFAULT_LOG_DIR, level="DEBUG", filename="demo.log", rotation="5 MB"
    )

    logger.debug("DEBUG：变量细节只进文件也无所谓,级别是可调的")
    logger.info("INFO：日志系统就绪")
    logger.warning("WARNING：显存快满了 (98%)")

    with log_time("模拟一个 epoch"):
        time.sleep(0.25)

    with Timer() as timer:
        time.sleep(0.25)
    logger.info("Timer 示例耗时：{:.3f} s", timer.elapsed)

    try:
        1 / 0
    except ZeroDivisionError:
        # 比 logger.exception 更省事：装饰器 @logger.catch 也能自动兜住
        logger.exception("ERROR：捕获异常并记录 traceback")

    logger.success("SUCCESS：这是 loguru 自带的级别,很好用")
    print(f"\n日志文件：{LOG_FILE}")
