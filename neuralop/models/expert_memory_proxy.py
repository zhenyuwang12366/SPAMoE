# -*- coding: utf-8 -*-
import gc
import torch
import copy
import contextlib
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple


class ExpertMemoryProxy:
    """
    透明代理：保证 forward_expert(expert_idx, x, **kwargs) ≡ experts[expert_idx](x, **kwargs)
    - 不处理任何索引映射；expert_idx 直接就是 experts 列表下标
    - 专家主权重常驻 CPU（eval + requires_grad=False）
    - 按需上卡（可半精度），仅前向；LRU 缓存、动态显存判断与 OOM 回退
    """

    def __init__(
        self,
        experts: List[torch.nn.Module],
        device: str | torch.device = "cuda",
        cache_size: int = 2,
        amp_dtype: Optional[torch.dtype] = torch.float16,
        convert_param_dtype_on_gpu: bool = True,
        safety_ratio: float = 1.20,
        measure_on_first_use: bool = True,
    ):
        self.device = torch.device(device)
        self.cache_size = int(cache_size)
        self.amp_dtype = amp_dtype
        self.convert_param_dtype_on_gpu = bool(convert_param_dtype_on_gpu)
        self.safety_ratio = float(safety_ratio)
        self.measure_on_first_use = bool(measure_on_first_use)

        self.cpu_experts: List[torch.nn.Module] = []
        self.gpu_cache: OrderedDict[int, torch.nn.Module] = OrderedDict()
        self.model_mem_est: Dict[int, int] = {}

        # ======= 主迁移与释放 =======
        # 注意：不要在遍历时修改原列表，先复制后遍历，遍历完再统一清空
        for m in list(experts):
            m_cpu = m.to("cpu", non_blocking=True)
            m_cpu.eval()
            for p in m_cpu.parameters():
                p.requires_grad_(False)

            # ★ 提前预置为普通属性（不是 buffer）
            if not hasattr(m_cpu, "ds_grads_remaining"):
                m_cpu.ds_grads_remaining = 0

            self.cpu_experts.append(m_cpu)

        # 释放原 experts 的引用（如果你希望尽快释放）
        try:
            experts.clear()
        except Exception:
            # 如果传入的是元组或其他不可变结构，忽略
            pass

        # 手动触发垃圾回收 + 清缓存
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"[ExpertMemoryProxy] init: experts={len(self.cpu_experts)}, "
            f"device={self.device}, cache_size={self.cache_size}, amp_dtype={self.amp_dtype}, "
            f"convert_param_dtype_on_gpu={self.convert_param_dtype_on_gpu}, safety_ratio={self.safety_ratio}, "
            f"measure_on_first_use={self.measure_on_first_use}"
        )

    # -------- 内部工具 --------

    @staticmethod
    def _null_ctx():
        return contextlib.nullcontext()

    def _amp_ctx(self, dtype: Optional[torch.dtype]):
        # 明确 device_type，避免在 CPU 场景行为不一致
        if dtype is not None:
            return torch.amp.autocast(device_type=self.device.type, dtype=dtype)
        return self._null_ctx()

    def _mem_info(self) -> Tuple[int, int]:
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            return int(free), int(total)
        # 非 CUDA 设备，给出一个“无限”空间的近似（不触发逐出逻辑）
        return 1 << 60, 1 << 60  # 1 exa-ish

    def _ensure_idx(self, idx: int) -> int:
        # 统一索引守卫：必须是 0..len-1 的整型
        if not isinstance(idx, (int,)):
            try:
                idx = int(idx)
            except Exception:
                raise IndexError(f"[ExpertMemoryProxy] idx={idx} 不是可转 int 的索引")
        n = len(self.cpu_experts)
        if not (0 <= idx < n):
            raise IndexError(f"[ExpertMemoryProxy] idx={idx} 越界（允许 0..{n-1}，实际 experts={n}）")
        return idx

    def _clone_to_device(self, m_cpu: torch.nn.Module) -> torch.nn.Module:
        m_gpu = copy.deepcopy(m_cpu)
        m_gpu.to(self.device, non_blocking=self.device.type=="cuda")
        m_gpu.eval()
        for p in m_gpu.parameters():
            p.requires_grad_(False)

        # ★ 依然只设“普通属性”，不要 register_buffer
        #   避免之后 DeepSpeed 在 backward 里做 `module.ds_grads_remaining = 0`
        #   时出现类型冲突
        if not hasattr(m_gpu, "ds_grads_remaining"):
            m_gpu.ds_grads_remaining = 0

        # 下面保持你的 dtype 转换逻辑（注意：只转浮点 buffer）
        if self.amp_dtype is not None and self.convert_param_dtype_on_gpu:
            for p in m_gpu.parameters():
                if p.is_floating_point():
                    p.data = p.data.to(self.amp_dtype)
            for name, b in m_gpu.named_buffers(recurse=True):
                if b.is_floating_point():
                    setattr(m_gpu, name, b.to(self.amp_dtype))

        return m_gpu

    def _estimate_model_mem_once(self, idx: int) -> int:
        idx = self._ensure_idx(idx)
        if self.device.type != "cuda":
            # 非 CUDA 设备：估一个参数/缓冲量级的近似
            m = self.cpu_experts[idx]
            bytes_params = sum(p.numel() * p.element_size() for p in m.parameters())
            bytes_bufs = sum(b.numel() * b.element_size() for b in m.buffers())
            est = int(1.3 * (bytes_params + bytes_bufs))
            self.model_mem_est[idx] = max(1, est)
            return self.model_mem_est[idx]

        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(self.device)
        m_gpu = self._clone_to_device(self.cpu_experts[idx])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        after = torch.cuda.memory_allocated(self.device)
        est = max(1, int(after - before))
        del m_gpu
        torch.cuda.empty_cache()
        self.model_mem_est[idx] = est
        return est

    def _get_est_mem(self, idx: int) -> int:
        idx = self._ensure_idx(idx)
        if idx in self.model_mem_est:
            return self.model_mem_est[idx]
        if self.measure_on_first_use:
            return self._estimate_model_mem_once(idx)
        m = self.cpu_experts[idx]
        bytes_params = sum(p.numel() * p.element_size() for p in m.parameters())
        bytes_bufs = sum(b.numel() * b.element_size() for b in m.buffers())
        est = int(1.3 * (bytes_params + bytes_bufs))
        self.model_mem_est[idx] = max(1, est)
        return est

    def _evict_until_fit(self, need_bytes: int) -> bool:
        if self.device.type != "cuda":
            # 非 CUDA 不做显存管理，视为始终 OK
            return True

        def ok():
            free, _ = self._mem_info()
            return free >= int(need_bytes * self.safety_ratio)

        if ok():
            return True
        while self.gpu_cache and not ok():
            _, old_m = self.gpu_cache.popitem(last=False)
            del old_m
            torch.cuda.empty_cache()
        return ok()

    def _admit(self, idx: int) -> Tuple[torch.nn.Module, bool]:
        """
        返回 (model_on_gpu, is_cached)
        - 命中缓存：直接复用
        - 未命中：判断显存→(缓存 or 临时) 上卡
        """
        idx = self._ensure_idx(idx)

        if idx in self.gpu_cache:
            m = self.gpu_cache.pop(idx)
            self.gpu_cache[idx] = m  # LRU 触发“最近使用”
            return m, True

        est = self._get_est_mem(idx)
        can_cache = self._evict_until_fit(est)
        m_gpu = self._clone_to_device(self.cpu_experts[idx])

        if can_cache:
            self.gpu_cache[idx] = m_gpu
            if len(self.gpu_cache) > self.cache_size:
                _, old_m = self.gpu_cache.popitem(last=False)
                del old_m
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            return m_gpu, True
        else:
            return m_gpu, False  # 临时加载，用完释放

    # -------- 对外 API：功能等价调用 --------

    @torch.no_grad()
    def forward_expert(self, expert_idx: int, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        保证功能等价：experts[expert_idx](x, **kwargs)
        仅做显存调度，不改索引绑定、不改前向语义
        """
        expert_idx = self._ensure_idx(expert_idx)
        x = x.to(self.device, non_blocking=True if self.device.type == "cuda" else False)
        m_gpu, cached = self._admit(expert_idx)
        try:
            with torch.inference_mode(), self._amp_ctx(self.amp_dtype):
                y = m_gpu(x, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and self.device.type == "cuda":
                # OOM 回退：清缓存 + 临时上卡再试一次
                self.clear_gpu_cache()
                torch.cuda.empty_cache()
                m_tmp = self._clone_to_device(self.cpu_experts[expert_idx])
                try:
                    with torch.inference_mode(), self._amp_ctx(self.amp_dtype):
                        y = m_tmp(x, **kwargs)
                finally:
                    del m_tmp
                    torch.cuda.empty_cache()
            else:
                raise
        finally:
            if not cached:
                del m_gpu
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return y

    @torch.no_grad()
    def forward_many(
        self,
        routed: Dict[int, torch.Tensor],
        fw_kwargs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        批量前向：{expert_idx -> x_sub} → {expert_idx -> y_sub}
        - 索引即身份，直接使用传入的 expert_idx
        - fw_kwargs 可为 None 或 {idx: {...}}，逐专家透传
        """
        outs: Dict[int, torch.Tensor] = {}
        kw = fw_kwargs or {}

        # 统一校验 routed / fw_kwargs key 边界
        n = len(self.cpu_experts)
        if n == 0 and (routed or kw):
            raise IndexError("[ExpertMemoryProxy] 当前没有任何专家，但收到了 routed/fw_kwargs 调用")

        bad_keys = [k for k in routed.keys() if not (isinstance(k, int) and 0 <= int(k) < n)]
        if bad_keys:
            raise IndexError(f"[ExpertMemoryProxy] routed keys 非法: {sorted(bad_keys)}，允许 0..{n-1}")
        if kw:
            bad_kw = [k for k in kw.keys() if not (isinstance(k, int) and 0 <= int(k) < n)]
            if bad_kw:
                raise IndexError(f"[ExpertMemoryProxy] fw_kwargs keys 非法: {sorted(bad_kw)}，允许 0..{n-1}")

        # 先跑缓存命中的
        pending: List[int] = []
        for idx, x in routed.items():
            idx = self._ensure_idx(idx)
            if idx in self.gpu_cache:
                m = self.gpu_cache.pop(idx)
                self.gpu_cache[idx] = m  # LRU: 最近使用
                x_dev = x.to(self.device, non_blocking=True if self.device.type == "cuda" else False)
                try:
                    with torch.inference_mode(), self._amp_ctx(self.amp_dtype):
                        outs[idx] = m(x_dev, **kw.get(idx, {}))
                except IndexError as e:
                    raise IndexError(
                        f"[ExpertMemoryProxy] expert idx={idx} 的前向内部触发 IndexError；"
                        f"x.shape={tuple(x_dev.shape)}，experts={n}"
                    ) from e
            else:
                pending.append(idx)

        if not pending:
            return outs

        # 统计待加载专家的显存需求
        need = [(i, self._get_est_mem(i)) for i in pending]
        i, total = 0, len(need)

        while i < total:
            free, _ = self._mem_info()
            budget = int(free / self.safety_ratio)
            if budget <= 0:
                self.clear_gpu_cache()
                free, _ = self._mem_info()
                budget = int(free / self.safety_ratio)
                if budget <= 0:
                    budget = need[i][1]  # 至少保证单个

            # 贪心装一批
            batch_ids: List[int] = []
            used = 0
            j = i
            while j < total and used + need[j][1] <= budget:
                batch_ids.append(need[j][0])
                used += need[j][1]
                j += 1
            if not batch_ids:
                batch_ids = [need[i][0]]
                j = i + 1

            loaded: List[Tuple[int, torch.nn.Module, bool]] = []
            try:
                # 上卡
                for idx in batch_ids:
                    idx = self._ensure_idx(idx)
                    m_gpu, cached = self._admit(idx)
                    loaded.append((idx, m_gpu, cached))

                # 前向
                for idx, m_gpu, cached in loaded:
                    x = routed[idx].to(self.device, non_blocking=True if self.device.type == "cuda" else False)
                    try:
                        with torch.inference_mode(), self._amp_ctx(self.amp_dtype):
                            outs[idx] = m_gpu(x, **kw.get(idx, {}))
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e) and self.device.type == "cuda":
                            outs.pop(idx, None)
                            self.clear_gpu_cache()
                            torch.cuda.empty_cache()
                            m_tmp = self._clone_to_device(self.cpu_experts[idx])
                            try:
                                with torch.inference_mode(), self._amp_ctx(self.amp_dtype):
                                    outs[idx] = m_tmp(x, **kw.get(idx, {}))
                            finally:
                                del m_tmp
                                torch.cuda.empty_cache()
                        else:
                            raise
                    except IndexError as e:
                        raise IndexError(
                            f"[ExpertMemoryProxy] expert idx={idx} 的前向内部触发 IndexError；"
                            f"x.shape={tuple(x.shape)}，experts={n}"
                        ) from e
            finally:
                # 释放临时上卡
                for idx, m_gpu, cached in loaded:
                    if not cached:
                        del m_gpu
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            i = j

        return outs

    # -------- 维护 --------
    def clear_gpu_cache(self):
        for _, m in self.gpu_cache.items():
            del m
        self.gpu_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def gpu_count(self) -> int:
        return len(self.gpu_cache)

    def cpu_count(self) -> int:
        return len(self.cpu_experts)

    def estimated_bytes(self, idx: int) -> int:
        """返回该专家上卡显存估计（字节），若未测量会触发一次测量/估计。"""
        return self._get_est_mem(idx)