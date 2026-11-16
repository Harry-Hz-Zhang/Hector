"""启动真实 hectord 的端到端用例（fixture 见 conftest.py）。"""

import subprocess
import sys
from pathlib import Path

import grpc
import pytest

from hector_client import HectorClient

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo.py"


def test_e2e_list_databases_empty(hectord):
    with HectorClient(hectord) as client:
        assert client.list_databases() == []


def test_e2e_demo_runs(hectord):
    result = subprocess.run(
        [sys.executable, str(DEMO), "--address", hectord],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "demo finished" in result.stdout
    with HectorClient(hectord) as client:
        assert client.list_databases() == []


def test_e2e_errors_surface_as_rpc_error(hectord):
    with HectorClient(hectord) as client:
        with pytest.raises(grpc.RpcError) as excinfo:
            client.list_tables("no_such_db")
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND
