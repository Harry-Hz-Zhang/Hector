#!/usr/bin/env bash
# 由 proto/hector/v1/hector.proto 生成 Python 存根到 clients/python/hector_client/_gen/（不入库）。
# 可在任意目录执行；默认使用 clients/python/.venv 中的解释器，可用环境变量 PYTHON 覆盖。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON:-${repo_root}/clients/python/.venv/bin/python}"
out_dir="clients/python/hector_client/_gen"

cd "${repo_root}"
rm -rf "${out_dir}"
mkdir -p "${out_dir}"
"${python_bin}" -m grpc_tools.protoc -I proto \
  --python_out="${out_dir}" \
  --grpc_python_out="${out_dir}" \
  proto/hector/v1/hector.proto
touch "${out_dir}/hector/__init__.py" "${out_dir}/hector/v1/__init__.py"
echo "generated ${out_dir}/hector/v1/hector_pb2.py and hector_pb2_grpc.py"
