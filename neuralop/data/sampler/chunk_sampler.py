import math
import random
from collections.abc import Iterator
from typing import Optional, TypeVar, List

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

_T_co = TypeVar("_T_co", covariant=True)

class ChunkDistributedSampler(Sampler[_T_co]):
    r"""
    分布式按块采样器：
    - 以 `chunk_size` 为粒度进行乱序（而非样本级乱序）；
    - 将打乱后的 chunk 等分到各 rank（必要时按 chunk 维度补齐或截断，使其整除）；
    - 在各自 rank 内，对每个 chunk 的样本索引做（可选）块内乱序，并按 `batch_size` 切分；
    - 这样 DataLoader 顺序取 batch 时，基本不会跨块，I/O 显著更平滑。

    参数
    ----
    dataset: Dataset
    num_replicas: 分布式进程数（world_size）
    rank: 当前进程 rank
    chunk_size: Zarr 第 0 维块大小（每块包含的样本数）
    batch_size: 每个 rank 的 batch 大小（注意：是 per-rank）
    shuffle: 是否对 chunk 级别打乱（默认 True）
    seed: 乱序种子（与 epoch 叠加）
    drop_last: True 时在块内对不满一个 batch 的尾部丢弃；False 时在块内循环补齐到整 batch
    intra_chunk_shuffle: 是否对每个 chunk 内部的样本索引做随机打乱（默认 True）
    """
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        *,
        chunk_size: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        intra_chunk_shuffle: bool = True,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank} (0..{num_replicas-1})")

        if batch_size > chunk_size:
            raise ValueError(
                f"batch_size({batch_size}) should be <= chunk_size({chunk_size}) "
                "to keep batches within a single chunk."
            )

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.chunk_size = int(chunk_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.intra_chunk_shuffle = bool(intra_chunk_shuffle)

        self.epoch = 0
        self.N = len(self.dataset)  # type: ignore[arg-type]
        self.num_chunks = math.ceil(self.N / self.chunk_size)

        # 为了保证各 rank 步数一致：把 chunk 列表补齐/截断为可被 num_replicas 整除
        if self.drop_last:
            self.num_chunks_total = (self.num_chunks // self.num_replicas) * self.num_replicas
        else:
            self.num_chunks_total = math.ceil(self.num_chunks / self.num_replicas) * self.num_replicas

        # 预估每块可产生的样本数量（按 batch 对齐）
        # 用于 __len__ 的近似/下界（实际 __iter__ 严格产出）
        full_batches_per_chunk = self.chunk_size // self.batch_size
        if self.drop_last:
            self.samples_per_chunk = full_batches_per_chunk * self.batch_size
        else:
            # 不丢尾：每块向上取整到 batch 的整数倍
            need = (-self.chunk_size) % self.batch_size
            self.samples_per_chunk = self.chunk_size + need

        # 近似每个 rank 的样本数（用于 __len__）
        chunks_per_rank = self.num_chunks_total // self.num_replicas
        self._len_estimate = chunks_per_rank * self.samples_per_chunk

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        # 注意：真实产出严格按 __iter__ 生成；这里给出稳定的一致长度
        return self._len_estimate

    def __iter__(self) -> Iterator[_T_co]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1) 构造 chunk id 列表并按需打乱
        chunk_ids = list(range(self.num_chunks))
        if self.shuffle:
            perm = torch.randperm(self.num_chunks, generator=g).tolist()
            chunk_ids = [chunk_ids[i] for i in perm]

        # 2) 补齐/截断到可被 num_replicas 整除
        if len(chunk_ids) < self.num_chunks_total:
            # pad：循环补到目标长度
            extra = self.num_chunks_total - len(chunk_ids)
            chunk_ids += (chunk_ids * math.ceil(extra / len(chunk_ids)))[:extra]
        else:
            chunk_ids = chunk_ids[: self.num_chunks_total]

        assert len(chunk_ids) % self.num_replicas == 0

        # 3) 均分到各 rank（关键：在“chunk 维度”做切片，保持块粒度的独立性）
        chunk_ids_rank = chunk_ids[self.rank :: self.num_replicas]

        # 4) 在当前 rank 内，逐块产出按 batch 对齐的索引
        indices_rank: List[int] = []
        for cid in chunk_ids_rank:
            lo = cid * self.chunk_size
            hi = min((cid + 1) * self.chunk_size, self.N)
            idxs = list(range(lo, hi))

            if self.intra_chunk_shuffle and len(idxs) > 1:
                # 在块内随机，但仍保持后续按 batch 切分
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]

            if self.drop_last:
                usable = (len(idxs) // self.batch_size) * self.batch_size
                idxs = idxs[:usable]
                # 若该块不足一批，直接跳过（典型发生在最后一块）
                if usable == 0:
                    continue
            else:
                # 补齐到整批（在块内循环补，避免跨块）
                need = (-len(idxs)) % self.batch_size
                if need:
                    idxs += idxs[:need]

            indices_rank.extend(idxs)

        # 5) 现在 indices_rank 已经是“按块分组且按 batch 对齐”的顺序；
        #    DataLoader 顺序取样时，每个 batch 会落在单一 chunk（或极少跨块，视补齐策略而定）
        return iter(indices_rank)