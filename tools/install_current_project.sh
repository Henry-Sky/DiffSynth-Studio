#!/usr/bin/env bash
# Reinstall this checkout in editable mode and verify that imports resolve here.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

cd "$project_dir"
"$python_bin" -m pip install -e .

module_path="$("$python_bin" -c 'import diffsynth; print(diffsynth.__file__)')"
expected_path="$project_dir/diffsynth/"
if [[ "$module_path" != "$expected_path"* ]]; then
    echo "错误：diffsynth 未指向当前项目：$module_path" >&2
    exit 1
fi

echo "环境已修复，当前项目模块：$module_path"
