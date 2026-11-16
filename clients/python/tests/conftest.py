"""端到端用例的 hectord fixture。

hectord 的日志全部写 stderr，且进程没有忽略 SIGPIPE。这里把 stderr 重定向到临时文件，
而不是管道：管道只读一行就不再读取时，日志写满缓冲区会阻塞服务线程；关闭读端则会让
hectord 被 SIGPIPE 杀死。启动后轮询日志文件，直到出现 `listening on <地址>`。
"""

import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HECTOR_BIN = REPO_ROOT / "build" / "debug" / "hectord"
STARTUP_TIMEOUT_S = 15.0
SHUTDOWN_TIMEOUT_S = 10.0
_LISTENING = re.compile(r"listening on (\S+)")


def _hector_bin():
    value = os.environ.get("HECTOR_BIN")
    return Path(value).resolve() if value else DEFAULT_HECTOR_BIN


def _stop(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture
def hectord(tmp_path):
    """以临时数据目录与系统分配的端口启动 hectord，返回 `host:port`。"""
    binary = _hector_bin()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        pytest.skip(f"hectord 不可用：{binary} 不存在或不可执行；请先构建或设置 HECTOR_BIN")

    log_path = tmp_path / "hectord.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [
                str(binary),
                f"--data_dir={tmp_path / 'data'}",
                "--listen_address=127.0.0.1:0",
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
    try:
        address = None
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while address is None:
            match = _LISTENING.search(log_path.read_text(errors="replace"))
            if match:
                address = match.group(1)
            elif proc.poll() is not None:
                pytest.fail(
                    f"hectord 启动失败，退出码 {proc.returncode}，日志：\n"
                    + log_path.read_text(errors="replace")
                )
            elif time.monotonic() > deadline:
                pytest.fail(
                    f"hectord 在 {STARTUP_TIMEOUT_S} 秒内没有输出 listening on，日志：\n"
                    + log_path.read_text(errors="replace")
                )
            else:
                time.sleep(0.05)
        yield address
    finally:
        _stop(proc)
