#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

cmake --preset macos-release
cmake --build --preset macos-release -j

ODE_PERF_SAMPLES="${ODE_PERF_SAMPLES:-20}" \
ODE_PERF_ITERATIONS="${ODE_PERF_ITERATIONS:-1000}" \
  ./build/macos-release/ode_perf_benchmark
