# -*- coding: utf-8 -*-
import gc
import torch
import copy
import contextlib
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple


class ExpertMemoryProxy:
    """
    Transparent proxy: ensures forward_expert(expert_idx, x, **kwargs) is equivalent to experts[expert_idx](x, **kwargs).
    - Does not remap indices; expert_idx is the experts list index directly.
    - Expert weights stay on CPU (eval + requires_grad=False).
    - Load to GPU on demand (optional half precision), forward only; LRU cache, dynamic VRAM checks and OOM fallback.
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

        self.training = True
        
        # ======= Main migration and release =======
        for m in list(experts):
            m_cpu = m.to("cpu", non_blocking=True)
            m_cpu.eval()
            for p in m_cpu.parameters():
                p.requires_grad_(False)

            # Ordinary attribute so frameworks do not treat it as a buffer
            if not hasattr(m_cpu, "ds_grads_remaining"):
                m_cpu.ds_grads_remaining = 0

            self.cpu_experts.append(m_cpu)

        # Drop references to original experts (optional, to free sooner)
        try:
            experts.clear()
        except Exception:
            pass

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"[ExpertMemoryProxy] init: experts={len(self.cpu_experts)}, "
            f"device={self.device}, cache_size={self.cache_size}, amp_dtype={self.amp_dtype}, "
            f"convert_param_dtype_on_gpu={self.convert_param_dtype_on_gpu}, safety_ratio={self.safety_ratio}, "
            f"measure_on_first_use={self.measure_on_first_use}"
        )

    # -------- Internal helpers --------

    @staticmethod
    def _null_ctx():
        return contextlib.nullcontext()

    def _amp_ctx(self, dtype: Optional[torch.dtype]):
        """
        AMP context: enable autocast only when dtype is not None.
        device_type matches the actual device (cuda/cpu).
        """
        if dtype is not None:
            return torch.amp.autocast(device_type=self.device.type, dtype=dtype)
        return self._null_ctx()

    def _fw_ctx(self):
        """
        Forward context:
        - Training (self.training=True and global grad enabled): allow gradients (nullcontext)
        - Eval: inference_mode for speed, no graph
        """
        if getattr(self, "training", False) and torch.is_grad_enabled():
            return self._null_ctx()
        return torch.inference_mode()

    def _mem_info(self) -> Tuple[int, int]:
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            return int(free), int(total)
        return 1 << 60, 1 << 60  # Non-CUDA: treat as unlimited

    def _ensure_idx(self, idx: int) -> int:
        if not isinstance(idx, (int,)):
            try:
                idx = int(idx)
            except Exception:
                raise IndexError(f"[ExpertMemoryProxy] idx={idx} is not convertible to int")
        n = len(self.cpu_experts)
        if not (0 <= idx < n):
            raise IndexError(f"[ExpertMemoryProxy] idx={idx} out of range (allowed 0..{n-1}, num experts={n})")
        return idx

    def _clone_to_device(self, m_cpu: torch.nn.Module) -> torch.nn.Module:
        m_gpu = copy.deepcopy(m_cpu)
        m_gpu.to(self.device, non_blocking=self.device.type == "cuda")
        m_gpu.eval()
        for p in m_gpu.parameters():
            p.requires_grad_(False)

        if not hasattr(m_gpu, "ds_grads_remaining"):
            m_gpu.ds_grads_remaining = 0

        # Cast floating point params/buffers to amp_dtype (e.g. fp16/bf16)
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
        Returns (model_on_gpu, is_cached)
        - Cache hit: reuse
        - Miss: check VRAM -> load to cache or temporary GPU
        """
        idx = self._ensure_idx(idx)

        if idx in self.gpu_cache:
            m = self.gpu_cache.pop(idx)
            self.gpu_cache[idx] = m  # LRU: mark most recently used
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
            return m_gpu, False  # Temporary load, release after use

    # -------- Public API: semantics-preserving forward (training allows grad; eval uses inference) --------

    def forward_expert(self, expert_idx: int, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Semantics match experts[expert_idx](x, **kwargs).
        Only VRAM scheduling; same index binding and forward meaning.
        - Training: gradients allowed (autograd)
        - Eval: inference_mode for speed
        """
        expert_idx = self._ensure_idx(expert_idx)
        x = x.to(self.device, non_blocking=True if self.device.type == "cuda" else False)
        m_gpu, cached = self._admit(expert_idx)
        try:
            with self._fw_ctx(), self._amp_ctx(self.amp_dtype):
                y = m_gpu(x, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and self.device.type == "cuda":
                # OOM fallback: clear cache + temporary GPU load and retry
                self.clear_gpu_cache()
                torch.cuda.empty_cache()
                m_tmp = self._clone_to_device(self.cpu_experts[expert_idx])
                try:
                    with self._fw_ctx(), self._amp_ctx(self.amp_dtype):
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

    def forward_many(
        self,
        routed: Dict[int, torch.Tensor],
        fw_kwargs: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Batched forward: {expert_idx -> x_sub} -> {expert_idx -> y_sub}
        - Indices are identities; use passed expert_idx as-is
        - fw_kwargs may be None or {idx: {...}} per expert
        - Training: gradients allowed; eval: inference_mode
        """
        outs: Dict[int, torch.Tensor] = {}
        kw = fw_kwargs or {}

        n = len(self.cpu_experts)
        if n == 0 and (routed or kw):
            raise IndexError("[ExpertMemoryProxy] no experts registered but routed/fw_kwargs was called")

        bad_keys = [k for k in routed.keys() if not (isinstance(k, int) and 0 <= int(k) < n)]
        if bad_keys:
            raise IndexError(f"[ExpertMemoryProxy] invalid routed keys: {sorted(bad_keys)}; allowed 0..{n-1}")
        if kw:
            bad_kw = [k for k in kw.keys() if not (isinstance(k, int) and 0 <= int(k) < n)]
            if bad_kw:
                raise IndexError(f"[ExpertMemoryProxy] invalid fw_kwargs keys: {sorted(bad_kw)}; allowed 0..{n-1}")

        # Run cache hits first
        pending: List[int] = []
        for idx, x in routed.items():
            idx = self._ensure_idx(idx)
            if idx in self.gpu_cache:
                m = self.gpu_cache.pop(idx)
                self.gpu_cache[idx] = m  # LRU: most recent
                x_dev = x.to(self.device, non_blocking=True if self.device.type == "cuda" else False)
                try:
                    with self._fw_ctx(), self._amp_ctx(self.amp_dtype):
                        outs[idx] = m(x_dev, **kw.get(idx, {}))
                except IndexError as e:
                    raise IndexError(
                        f"[ExpertMemoryProxy] expert idx={idx} forward raised IndexError; "
                        f"x.shape={tuple(x_dev.shape)}, num experts={n}"
                    ) from e
            else:
                pending.append(idx)

        if not pending:
            return outs

        # VRAM needs for experts still to load
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
                    budget = need[i][1]  # At least one expert

            # Greedy batch packing
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
                # Move to GPU
                for idx in batch_ids:
                    idx = self._ensure_idx(idx)
                    m_gpu, cached = self._admit(idx)
                    loaded.append((idx, m_gpu, cached))

                # Forward
                for idx, m_gpu, cached in loaded:
                    x = routed[idx].to(self.device, non_blocking=True if self.device.type == "cuda" else False)
                    try:
                        with self._fw_ctx(), self._amp_ctx(self.amp_dtype):
                            outs[idx] = m_gpu(x, **kw.get(idx, {}))
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e) and self.device.type == "cuda":
                            outs.pop(idx, None)
                            self.clear_gpu_cache()
                            torch.cuda.empty_cache()
                            m_tmp = self._clone_to_device(self.cpu_experts[idx])
                            try:
                                with self._fw_ctx(), self._amp_ctx(self.amp_dtype):
                                    outs[idx] = m_tmp(x, **kw.get(idx, {}))
                            finally:
                                del m_tmp
                                torch.cuda.empty_cache()
                        else:
                            raise
                    except IndexError as e:
                        raise IndexError(
                            f"[ExpertMemoryProxy] expert idx={idx} forward raised IndexError; "
                            f"x.shape={tuple(x.shape)}, num experts={n}"
                        ) from e
            finally:
                # Release temporary GPU loads
                for idx, m_gpu, cached in loaded:
                    if not cached:
                        del m_gpu
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            i = j

        return outs

    # -------- Maintenance --------
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
        """Estimated GPU bytes for this expert; may trigger one measurement/estimate if unknown."""
        return self._get_est_mem(idx)
