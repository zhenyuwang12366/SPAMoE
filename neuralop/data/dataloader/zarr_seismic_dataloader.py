# dataloader_build.py
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from ..sampler.chunk_sampler import ChunkDistributedSampler

def worker_init_reopen_zarr(_):
    # Zarr may be lazily opened in Dataset.__getitem__; here tune blosc threads, etc.
    try:
        import numcodecs.blosc as blosc
        blosc.set_nthreads(max(1, os.cpu_count() // 2))
    except Exception:
        pass

def build_loaders(
    args, config,
    train_dataset_with_transform,
    val_dataset_with_transform,
    chunks,
    world_size=1, local_rank=0
):
    is_dist = bool(getattr(args, "distributed", False))
    per_rank_bs = int(config.batch_size)
    per_rank_test_bs = int(config.test_batch_size)

    if is_dist:
        if train_dataset_with_transform is not None:
            train_num_workers = max(0, args.num_workers // 2)

            # Training: chunk-wise distributed sampler (batches stay within one chunk)
            train_chunk = chunks
            print(f"[DEBUG] zarr use chunk: {train_chunk}")
            train_sampler = ChunkDistributedSampler(
                train_dataset_with_transform,
                num_replicas=world_size,
                rank=local_rank,
                chunk_size=train_chunk,
                batch_size=per_rank_bs,
                shuffle=True,
                drop_last=True,
                intra_chunk_shuffle=True,
                seed=args.seed,
            )
            train_loader = DataLoader(
                train_dataset_with_transform,
                sampler=train_sampler,
                batch_size=per_rank_bs,           # Must match sampler batch_size (per-rank)
                shuffle=False,
                num_workers=train_num_workers,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=train_num_workers > 0,
                worker_init_fn=worker_init_reopen_zarr,
                multiprocessing_context="forkserver",
            )

        # Validation: sequential distributed sampling (ChunkDistributedSampler(shuffle=False) is another option)
        val_num_workers = train_num_workers if train_dataset_with_transform is not None else max(0, args.num_workers)
        val_sampler = DistributedSampler(
            val_dataset_with_transform,
            num_replicas=world_size,
            rank=local_rank,
            drop_last=False,
            shuffle=False,
        )
        val_loader = DataLoader(
            val_dataset_with_transform,
            sampler=val_sampler,
            batch_size=per_rank_test_bs,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=val_num_workers > 0,
            worker_init_fn=worker_init_reopen_zarr,
            multiprocessing_context="forkserver",
        )
        return train_loader, val_loader, train_sampler, val_sampler
    else:
        # Single process: still use chunk sampling (less I/O) while keeping training randomness
        train_num_workers = max(0, args.num_workers)
        train_chunk = chunks

        if train_dataset_with_transform is not None:
            train_sampler = ChunkDistributedSampler(
                train_dataset_with_transform,
                num_replicas=1,
                rank=0,
                chunk_size=train_chunk,
                batch_size=per_rank_bs,
                shuffle=True,
                drop_last=True,
                intra_chunk_shuffle=True,
                seed=42,
            )
            train_loader = DataLoader(
                train_dataset_with_transform,
                sampler=train_sampler,
                batch_size=per_rank_bs,
                shuffle=False,
                num_workers=train_num_workers,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=train_num_workers > 0,
                worker_init_fn=worker_init_reopen_zarr,
            )

        val_loader = DataLoader(
            val_dataset_with_transform,
            batch_size=per_rank_test_bs,
            shuffle=False,
            num_workers=train_num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=train_num_workers > 0,
            worker_init_fn=worker_init_reopen_zarr,
        )
        return train_loader, val_loader, train_sampler, None