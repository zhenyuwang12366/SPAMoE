#!/usr/bin/env bash

find_conda_sh() {
  local candidate=""

  if [[ -n "${CONDA_EXE:-}" ]]; then
    candidate="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)/etc/profile.d/conda.sh"
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_base=""
    conda_base="$(conda info --base 2>/dev/null || true)"
    candidate="${conda_base}/etc/profile.d/conda.sh"
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi

  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "/root/miniconda3/etc/profile.d/conda.sh" \
    "/data1/apps/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

activate_conda_env() {
  local env_name="${1:-}"
  if [[ -z "${env_name}" ]]; then
    return 0
  fi

  if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" ]]; then
    return 0
  fi

  local conda_sh=""
  if ! conda_sh="$(find_conda_sh)"; then
    echo "[WARN] 未找到 conda.sh，继续使用当前 shell 环境。" >&2
    return 0
  fi

  # shellcheck disable=SC1090
  . "${conda_sh}"
  conda activate "${env_name}"
}

normalize_family_name() {
  local raw_family="${1:-}"
  case "${raw_family}" in
    style_a) printf '%s\n' "style_style_a" ;;
    style_b) printf '%s\n' "style_style_b" ;;
    *) printf '%s\n' "${raw_family}" ;;
  esac
}
