#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
用法:
  bash school_queue_after_current.sh --tasks_file pending_tasks.txt [选项]

功能:
  1. 轮询当前正在运行的 school_local.sh 任务
  2. 等这些任务全部结束后
  3. 再按批次启动 tasks_file 里的后续任务

tasks_file 格式:
  - 每行一条完整命令
  - 支持空行和 # 注释
  - 建议在每行里显式写 CUDA_VISIBLE_DEVICES，避免 GPU 冲突

示例:
  bash school_queue_after_current.sh \
    --tasks_file pending_tasks.example.txt \
    --poll_seconds 60 \
    --batch_size 2

常用选项:
  --tasks_file PATH       待执行任务列表文件
  --wait_pattern TEXT     用于匹配“当前正在跑的任务”的进程关键字
                          默认: bash school_local.sh
  --wait_until_count N    匹配到的运行中进程数 <= N 时开始后续任务，默认 0
  --poll_seconds N        轮询间隔，默认 60 秒
  --batch_size N          每批并行启动多少个任务，默认 2
EOF
}

TASKS_FILE=""
WAIT_PATTERN="bash school_local.sh"
WAIT_UNTIL_COUNT=0
POLL_SECONDS=60
BATCH_SIZE=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks_file) TASKS_FILE="$2"; shift 2 ;;
    --wait_pattern) WAIT_PATTERN="$2"; shift 2 ;;
    --wait_until_count) WAIT_UNTIL_COUNT="$2"; shift 2 ;;
    --poll_seconds) POLL_SECONDS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${TASKS_FILE}" ]]; then
  echo "必须提供 --tasks_file。" >&2
  usage
  exit 1
fi

if [[ ! -f "${TASKS_FILE}" ]]; then
  echo "任务文件不存在: ${TASKS_FILE}" >&2
  exit 1
fi

if ! [[ "${WAIT_UNTIL_COUNT}" =~ ^[0-9]+$ && "${POLL_SECONDS}" =~ ^[0-9]+$ && "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "wait_until_count / poll_seconds / batch_size 必须是合法整数。" >&2
  exit 1
fi

count_running_tasks() {
  ps -axo pid=,command= | awk -v pat="${WAIT_PATTERN}" -v self="$$" '
    index($0, pat) && $1 != self { count += 1 }
    END { print count + 0 }
  '
}

print_running_tasks() {
  ps -axo pid=,command= | awk -v pat="${WAIT_PATTERN}" -v self="$$" '
    index($0, pat) && $1 != self { print }
  '
}

TASKS=()
while IFS= read -r line || [[ -n "${line}" ]]; do
  case "${line}" in
    ''|[[:space:]]*\#*) continue ;;
  esac
  TASKS+=("${line}")
done < "${TASKS_FILE}"

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "任务文件里没有可执行命令: ${TASKS_FILE}" >&2
  exit 1
fi

echo "等待中的进程关键字: ${WAIT_PATTERN}"
echo "开始阈值: 运行中任务数 <= ${WAIT_UNTIL_COUNT}"
echo "轮询间隔: ${POLL_SECONDS}s"
echo "后续任务总数: ${#TASKS[@]}"
echo "每批并行数: ${BATCH_SIZE}"

while true; do
  running_count="$(count_running_tasks)"
  timestamp="$(date '+%F %T')"
  echo "[${timestamp}] 当前匹配到 ${running_count} 个运行中任务"
  if (( running_count <= WAIT_UNTIL_COUNT )); then
    break
  fi
  print_running_tasks || true
  sleep "${POLL_SECONDS}"
done

echo "当前任务已满足启动条件，开始执行后续队列。"

task_index=0
while (( task_index < ${#TASKS[@]} )); do
  PIDS=()
  CMDS=()
  launched=0

  while (( task_index < ${#TASKS[@]} && launched < BATCH_SIZE )); do
    cmd="${TASKS[$task_index]}"
    echo "[launch] ${cmd}"
    bash -lc "${cmd}" &
    PIDS+=("$!")
    CMDS+=("${cmd}")
    task_index=$((task_index + 1))
    launched=$((launched + 1))
    sleep 2
  done

  batch_failed=0
  i=0
  while (( i < ${#PIDS[@]} )); do
    pid="${PIDS[$i]}"
    cmd="${CMDS[$i]}"
    if wait "${pid}"; then
      echo "[done] ${cmd}"
    else
      status=$?
      echo "[fail] exit=${status} :: ${cmd}" >&2
      batch_failed=1
    fi
    i=$((i + 1))
  done

  if (( batch_failed != 0 )); then
    echo "检测到失败任务，停止继续提交后续批次。" >&2
    exit 1
  fi
done

echo "所有后续任务已执行完成。"
