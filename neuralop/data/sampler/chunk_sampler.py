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
    Distributed chunk-wise sampler:
    - Shuffles at `chunk_size` granularity (not per-sample);
    - Splits shuffled chunks across ranks (pad or trim along chunks so counts divide by num_replicas);
    - Within each rank, optionally shuffles indices inside each chunk and groups them by `batch_size`;
    - Batches from DataLoader then rarely straddle chunks, smoothing I/O.

    Parameters
    ----------
    dataset: Dataset
    num_replicas: number of distributed processes (world_size)
    rank: current process rank
    chunk_size: Zarr chunk length along dimension 0 (samples per chunk)
    batch_size: per-rank batch size
    shuffle: shuffle chunk order (default True)
    seed: shuffle seed (combined with epoch)
    drop_last: if True, drop tail of chunk that does not fill a batch; if False, wrap within chunk to full batches
    intra_chunk_shuffle: shuffle sample indices inside each chunk (default True)
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

        # Keep per-rank step counts aligned: pad/trim chunk list to multiple of num_replicas
        if self.drop_last:
            self.num_chunks_total = (self.num_chunks // self.num_replicas) * self.num_replicas
        else:
            self.num_chunks_total = math.ceil(self.num_chunks / self.num_replicas) * self.num_replicas

        # Expected samples per chunk (batch-aligned); used for __len__ estimate (__iter__ is authoritative)
        full_batches_per_chunk = self.chunk_size // self.batch_size
        if self.drop_last:
            self.samples_per_chunk = full_batches_per_chunk * self.batch_size
        else:
            # keep tail: round each chunk up to a multiple of batch_size
            need = (-self.chunk_size) % self.batch_size
            self.samples_per_chunk = self.chunk_size + need

        # Approximate samples per rank (for __len__)
        chunks_per_rank = self.num_chunks_total // self.num_replicas
        self._len_estimate = chunks_per_rank * self.samples_per_chunk

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        # Note: actual yield is defined by __iter__; this returns a stable consistent length
        return self._len_estimate

    def __iter__(self) -> Iterator[_T_co]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1) Build chunk id list and optionally shuffle
        chunk_ids = list(range(self.num_chunks))
        if self.shuffle:
            perm = torch.randperm(self.num_chunks, generator=g).tolist()
            chunk_ids = [chunk_ids[i] for i in perm]

        # 2) Pad/trim to multiple of num_replicas
        if len(chunk_ids) < self.num_chunks_total:
            # pad by cycling chunk_ids
            extra = self.num_chunks_total - len(chunk_ids)
            chunk_ids += (chunk_ids * math.ceil(extra / len(chunk_ids)))[:extra]
        else:
            chunk_ids = chunk_ids[: self.num_chunks_total]

        assert len(chunk_ids) % self.num_replicas == 0

        # 3) Split chunks across ranks (slice along chunk dimension keeps chunk locality)
        chunk_ids_rank = chunk_ids[self.rank :: self.num_replicas]

        # 4) Within this rank, emit batch-aligned indices per chunk
        indices_rank: List[int] = []
        for cid in chunk_ids_rank:
            lo = cid * self.chunk_size
            hi = min((cid + 1) * self.chunk_size, self.N)
            idxs = list(range(lo, hi))

            if self.intra_chunk_shuffle and len(idxs) > 1:
                # shuffle inside chunk while still grouping into batches afterward
                perm = torch.randperm(len(idxs), generator=g).tolist()
                idxs = [idxs[i] for i in perm]

            if self.drop_last:
                usable = (len(idxs) // self.batch_size) * self.batch_size
                idxs = idxs[:usable]
                # Chunk smaller than one batch: skip (typical for the last chunk)
                if usable == 0:
                    continue
            else:
                # pad to full batch by wrapping within chunk (no cross-chunk reads)
                need = (-len(idxs)) % self.batch_size
                if need:
                    idxs += idxs[:need]

            indices_rank.extend(idxs)

        # 5) indices_rank is grouped by chunk and batch-aligned;
        #    sequential DataLoader iteration keeps each batch in a single chunk (rare cross-chunk cases depend on padding)
        return iter(indices_rank)