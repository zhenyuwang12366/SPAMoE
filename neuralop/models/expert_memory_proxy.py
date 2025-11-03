# -*- coding: utf-8 -*-
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
        amp_dtype: Optional[torch.dtype] = torch.float16,  # 可设为 torch.bfloat16 或 None
        convert_param_dtype_on_gpu: bool = True,           # 上卡后将权重/缓冲也转 dtype 进一步省显存
        safety_ratio: float = 1.20,                        # 动态显存判断的安全系数
        measure_on_first_use: bool = True,                 # 首次精测该专家的上卡显存占用
    ):
        self.device = torch.device(device)
        self.cache_size = int(cache_size)
        self.amp_dtype = amp_dtype
        self.convert_param_dtype_on_gpu = bool(convert_param_dtype_on_gpu)
        self.safety_ratio = float(safety_ratio)
        self.measure_on_first_use = bool(measure_on_first_use)

        # CPU 常驻（不改变 experts 顺序/绑定）
        self.cpu_experts: List[torch.nn.Module] = []
        for m in experts:
            m = m.cpu().copy()
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.cpu_experts.append(m)
        
        # GPU LRU 缓存：idx -> model
        self.gpu_cache: OrderedDict[int, torch.nn.Module] = OrderedDict()
        # 每个专家上卡显存估计（字节）
        self.model_mem_est: Dict[int, int] = {}

    # -------- 内部工具 --------

    @staticmethod
    def _amp_ctx(dtype: Optional[torch.dtype]):
        return torch.cuda.amp.autocast(dtype=dtype) if dtype is not None else contextlib.nullcontext()

    def _mem_info(self) -> Tuple[int, int]:
        free, total = torch.cuda.mem_get_info(self.device)
        return int(free), int(total)

    def _clone_to_device(self, m_cpu: torch.nn.Module) -> torch.nn.Module:
        # 用 deepcopy，完全保留模块属性；避免依赖空构造器
        m_gpu = copy.deepcopy(m_cpu)
        m_gpu.to(self.device)
        m_gpu.eval()
        for p in m_gpu.parameters():
            p.requires_grad_(False)
        if self.amp_dtype is not None and self.convert_param_dtype_on_gpu:
            for p in m_gpu.parameters():
                if p.is_floating_point():
                    p.data = p.data.to(self.amp_dtype)
            for name, b in m_gpu.named_buffers(recurse=True):
                if b.is_floating_point():
                    setattr(m_gpu, name, b.to(self.amp_dtype))
        return m_gpu

    def _estimate_model_mem_once(self, idx: int) -> int:
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(self.device)
        m_gpu = self._clone_to_device(self.cpu_experts[idx])
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated(self.device)
        est = max(1, int(after - before))
        del m_gpu
        torch.cuda.empty_cache()
        self.model_mem_est[idx] = est
        return est

    def _get_est_mem(self, idx: int) -> int:
        if idx in self.model_mem_est:
            return self.model_mem_est[idx]
        if self.measure_on_first_use:
            return self._estimate_model_mem_once(idx)
        m = self.cpu_experts[idx]
        bytes_params = sum(p.numel() * p.element_size() for p in m.parameters())
        bytes_bufs   = sum(b.numel() * b.element_size() for b in m.buffers())
        est = int(1.3 * (bytes_params + bytes_bufs))
        self.model_mem_est[idx] = max(1, est)
        return est

    def _evict_until_fit(self, need_bytes: int) -> bool:
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
        x = x.to(self.device, non_blocking=True)
        m_gpu, cached = self._admit(expert_idx)
        try:
            with torch.inference_mode(), self._amp_ctx(self.amp_dtype):
                y = m_gpu(x, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
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

        # 先跑缓存命中的
        pending = []
        for idx, x in routed.items():
            if idx in self.gpu_cache:
                m = self.gpu_cache.pop(idx); self.gpu_cache[idx] = m
                x_dev = x.to(self.device, non_blocking=True)
                with torch.inference_mode(), self._amp_ctx(self.amp_dtype):
                    outs[idx] = m(x_dev, **kw.get(idx, {}))
            else:
                pending.append(idx)

        if not pending:
            return outs

        need = [(i, self._get_est_mem(i)) for i in pending]
        i, n = 0, len(need)

        while i < n:
            free, _ = self._mem_info()
            budget = int(free / self.safety_ratio)
            if budget <= 0:
                self.clear_gpu_cache()
                free, _ = self._mem_info()
                budget = int(free / self.safety_ratio)
                if budget <= 0:
                    budget = need[i][1]  # 至少保证单个

            # 贪心装一批
            batch_ids, used = [], 0
            j = i
            while j < n and used + need[j][1] <= budget:
                batch_ids.append(need[j][0])
                used += need[j][1]
                j += 1
            if not batch_ids:
                batch_ids = [need[i][0]]
                j = i + 1

            loaded = []
            try:
                for idx in batch_ids:
                    m_gpu, cached = self._admit(idx)
                    loaded.append((idx, m_gpu, cached))

                for idx, m_gpu, cached in loaded:
                    x = routed[idx].to(self.device, non_blocking=True)
                    try:
                        with torch.inference_mode(), self._amp_ctx(self.amp_dtype):
                            outs[idx] = m_gpu(x, **kw.get(idx, {}))
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            outs.pop(idx, None)
                            self.clear_gpu_cache(); torch.cuda.empty_cache()
                            m_tmp = self._clone_to_device(self.cpu_experts[idx])
                            try:
                                with torch.inference_mode(), self._amp_ctx(self.amp_dtype):
                                    outs[idx] = m_tmp(x, **kw.get(idx, {}))
                            finally:
                                del m_tmp
                                torch.cuda.empty_cache()
                        else:
                            raise
            finally:
                for idx, m_gpu, cached in loaded:
                    if not cached:
                        del m_gpu
                torch.cuda.empty_cache()

            i = j

        return outs

    # -------- 维护 --------
    def clear_gpu_cache(self):
        for _, m in self.gpu_cache.items():
            del m
        self.gpu_cache.clear()
        torch.cuda.empty_cache()

    def gpu_count(self) -> int:
        return len(self.gpu_cache)

    def cpu_count(self) -> int:
        return len(self.cpu_experts)

    def estimated_bytes(self, idx: int) -> int:
        """返回该专家上卡显存估计（字节），若未测量会触发一次测量/估计。"""
        return self._get_est_mem(idx)