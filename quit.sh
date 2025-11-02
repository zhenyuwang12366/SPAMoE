#!/bin/bash
# ============================================
# 自动清理 Python 训练残留与系统缓存脚本
# 适用于 DataLoader/Zarr 内存爆炸后的训练前环境重置
# ============================================

echo "=== 清理开始: $(date '+%F %T') ==="

# 1. 结束所有 Python 进程（避免旧训练残留）
echo "[1/6] Killing python processes..."
pkill -9 -u $(whoami) -f python 2>/dev/null
sleep 2

# 2. 清理共享内存段与临时缓存（DataLoader 多进程遗留）
echo "[2/6] Cleaning /dev/shm and /tmp..."
rm -rf /dev/shm/*
rm -rf /tmp/pymp*
rm -rf /tmp/torch_*
rm -rf /tmp/tmp*
sleep 1

# 3. 释放系统页缓存（需要 root 权限）
if [ "$(id -u)" -eq 0 ]; then
  echo "[3/6] Dropping OS caches..."
  sync
  echo 3 > /proc/sys/vm/drop_caches
else
  echo "[3/6] 跳过 drop_caches (需要 root 权限)"
fi

# 4. 清理孤立的共享内存段与信号量（慎用）
echo "[4/6] Removing IPC remnants..."
ipcs -m | grep $(whoami) | awk '{print $2}' | xargs -I{} ipcrm -m {} 2>/dev/null
ipcs -s | grep $(whoami) | awk '{print $2}' | xargs -I{} ipcrm -s {} 2>/dev/null

# 5. 清理 GPU 缓存（若使用 CUDA）
if command -v nvidia-smi &>/dev/null; then
  echo "[5/6] GPU 状态:"
  nvidia-smi
else
  echo "[5/6] 未检测到 nvidia-smi"
fi

# 6. 打印当前内存使用情况
echo "[6/6] 当前内存状态:"
free -h | awk 'NR<=2{print}'
echo "=== 清理完成 ==="
