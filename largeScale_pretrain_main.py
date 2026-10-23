import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from configs import Config
from channel_config import STANDARD_64_CHANNELS

from dataset import collate_fn
from lance_dataset import LanceEEGPretrainDataset
from prefetch_loader import PrefetchLoader

from model import EEGPretrainModel
from losses import BalancedBandTokenTrajectoryLoss
from trainer import train_one_epoch, validate_one_epoch
from masking import generate_block_mask, apply_token_mask
from utils import set_seed, ensure_dir_for_file

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Sampler
import math
import random
import numpy as np
import torch.distributed as dist

from lance_dataset import PartAwareDistributedBatchSampler


import torch.multiprocessing as mp

from ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    barrier,
)


from lance_dataset import (
    LanceEEGPretrainDataset,
    MultiLanceEEGPretrainDataset,
    load_storage_options,
    discover_lance_part_paths,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_ddp():
    """
    初始化 DDP。
    用 torchrun 启动时，会自动设置这些环境变量：
    RANK, WORLD_SIZE, LOCAL_RANK
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

        device = torch.device("cuda", local_rank)
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distributed = False

    return distributed, rank, world_size, local_rank, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0



class BatchedSubset(Dataset):
    """
    保留底层 dataset.__getitems__ 批量读取能力的子集包装器。
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[int(idx)]]

    def __getitems__(self, idxs):
        mapped_indices = [self.indices[int(i)] for i in idxs]
        if hasattr(self.dataset, "__getitems__"):
            return self.dataset.__getitems__(mapped_indices)
        return [self.dataset[i] for i in mapped_indices]




def compute_shape_params(cfg):
    patch_len = int(round(cfg.data.patch_seconds * cfg.data.target_sfreq))
    patch_stride = int(round(cfg.data.patch_stride_seconds * cfg.data.target_sfreq))
    total_points = int(round(cfg.data.clip_seconds * cfg.data.target_sfreq))

    num_time_patches = int(
        np.floor((total_points - patch_len) / patch_stride) + 1
    )

    if num_time_patches <= 0:
        raise ValueError(
            f"num_time_patches <= 0. "
            f"Please check clip_seconds={cfg.data.clip_seconds}, "
            f"patch_seconds={cfg.data.patch_seconds}, "
            f"target_sfreq={cfg.data.target_sfreq}"
        )

    num_tokens = cfg.data.n_channels * num_time_patches
    frames_per_patch = int(round(cfg.data.patch_seconds / cfg.data.target_frame_seconds))
    n_bands = len(cfg.data.band_defs)

    return {
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "total_points": total_points,
        "num_time_patches": num_time_patches,
        "num_tokens": num_tokens,
        "frames_per_patch": frames_per_patch,
        "n_bands": n_bands,
    }


def build_batch_token_mask(batch_size, num_tokens, cfg, device):
    masks = []
    for _ in range(batch_size):
        mask = generate_block_mask(
            num_tokens=num_tokens,
            mask_ratio=cfg.mask.mask_ratio,
        )
        masks.append(mask)

    return torch.tensor(
        np.stack(masks, axis=0),
        dtype=torch.float32,
        device=device,
    )


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.device.type == "cpu":
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def visualize_train_reconstruction(
    model,
    loader,
    cfg,
    device,
    save_dir,
    max_samples=4,
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    token_inputs = batch["token_inputs"]
    targets = batch["targets"]
    token_channel_indices = batch["token_channel_indices"]
    token_time_indices = batch["token_time_indices"]
    token_valid_mask = batch["token_valid_mask"]

    B, S, _ = token_inputs.shape

    token_mask = build_batch_token_mask(B, S, cfg, device)
    masked_inputs = apply_token_mask(tokens=token_inputs, token_mask=token_mask)

    pred_norm = model(
        token_inputs=masked_inputs,
        token_channel_indices=token_channel_indices,
        token_time_indices=token_time_indices,
    )

    pred = pred_norm.detach().cpu().numpy()
    target = targets.detach().cpu().numpy()
    token_mask_np = token_mask.detach().cpu().numpy()
    token_valid_np = token_valid_mask.detach().cpu().numpy()
    token_ch_np = token_channel_indices.detach().cpu().numpy()
    token_t_np = token_time_indices.detach().cpu().numpy()

    band_names = [x[0] for x in cfg.data.band_defs]
    num_to_plot = min(max_samples, B)

    for b in range(num_to_plot):
        valid_indices = np.where(token_valid_np[b] > 0.5)[0]
        masked_valid_indices = np.where(
            (token_valid_np[b] > 0.5) & (token_mask_np[b] > 0.5)
        )[0]
        unmasked_valid_indices = np.where(
            (token_valid_np[b] > 0.5) & (token_mask_np[b] < 0.5)
        )[0]

        if len(valid_indices) == 0:
            print(f"[Visualization] sample {b}: no valid token, skip.")
            continue
        if len(masked_valid_indices) == 0:
            print(f"[Visualization] sample {b}: no masked valid token, skip.")
            continue
        if len(unmasked_valid_indices) == 0:
            print(f"[Visualization] sample {b}: no unmasked valid token, skip.")
            continue

        masked_token_idx = int(np.random.choice(masked_valid_indices))
        unmasked_token_idx = int(np.random.choice(unmasked_valid_indices))

        examples = [("masked", masked_token_idx), ("unmasked", unmasked_token_idx)]

        F = target.shape[2]
        K = target.shape[3]
        x = np.arange(K)

        fig, axes = plt.subplots(
            nrows=F,
            ncols=2,
            figsize=(12, 2.2 * F),
            squeeze=False,
        )

        for col, (case_name, tok_idx) in enumerate(examples):
            ch_idx = int(token_ch_np[b, tok_idx])
            time_idx = int(token_t_np[b, tok_idx])
            ch_name = STANDARD_64_CHANNELS[ch_idx]

            for f in range(F):
                ax = axes[f][col]
                ax.plot(x, target[b, tok_idx, f], marker="o", linewidth=2, label="target")
                ax.plot(x, pred[b, tok_idx, f], marker="x", linewidth=2, label="pred")
                ax.set_title(
                    f"{case_name} | token={tok_idx} | {ch_name} | patch={time_idx} | {band_names[f]}"
                )
                ax.set_xlabel("frame inside patch")
                ax.set_ylabel("log band power")
                ax.grid(True, alpha=0.3)
                if f == 0:
                    ax.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"train_reconstruction_sample_{b}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        if is_main_process():
            print(f"[Visualization] saved: {save_path}")


def main():
    cfg = Config()

    distributed, rank, world_size, local_rank, device = setup_ddp()

    set_seed(cfg.train.seed + rank)

    if is_main_process():
        print("Using device:", device)
        print("Distributed:", distributed)
        print("rank:", rank)
        print("world_size:", world_size)
        print("local_rank:", local_rank)
        print("per-GPU batch size:", cfg.train.batch_size)
        print("global batch size:", cfg.train.batch_size * world_size)

    if is_main_process():
        print("\n===== Loading Processed EEG Samples From Lance =====")

    if cfg.data.use_multi_lance_parts:
        storage_options = load_storage_options(cfg.data.storage_options_path)

        part_paths = discover_lance_part_paths(
            root_path=cfg.data.lance_parts_root,
            storage_options=storage_options,
            glob_pattern=cfg.data.lance_part_glob,
        )

        print(f"Found {len(part_paths)} Lance parts.")
        print("First Lance part:", part_paths[0])
        print("Last Lance part: ", part_paths[-1])

        dataset = MultiLanceEEGPretrainDataset(
            lance_paths=part_paths,
            storage_options=storage_options,
        )
    else:
        dataset = LanceEEGPretrainDataset(
            lance_path=cfg.data.lance_path,
        )

    if is_main_process():
        print("dataset size:", len(dataset))

    shape_params = compute_shape_params(cfg)

    patch_len = shape_params["patch_len"]
    num_time_patches = shape_params["num_time_patches"]
    frames_per_patch = shape_params["frames_per_patch"]
    n_bands = shape_params["n_bands"]

    if is_main_process():
        print("\n===== Shape Params =====")
        for k, v in shape_params.items():
            print(f"{k}: {v}")

    if is_main_process():
        print("\n===== Loading Processed EEG Samples From Lance =====")


    if len(dataset) < 2:
        raise ValueError(
            "Too few samples. Need at least 2 for train/val split. "
            "Please regenerate Lance with shorter clip_seconds or use longer recordings."
        )

    s0 = dataset[0]

    if is_main_process():
        print("\nFirst Lance sample:")
        print("  token_inputs:          ", tuple(s0["token_inputs"].shape))
        print("  targets:               ", tuple(s0["targets"].shape))
        print("  token_channel_indices: ", tuple(s0["token_channel_indices"].shape))
        print("  token_time_indices:    ", tuple(s0["token_time_indices"].shape))
        print("  token_valid_mask:      ", tuple(s0["token_valid_mask"].shape))
        print("  channel_valid_mask:    ", tuple(s0["channel_valid_mask"].shape))
        print("  valid channels:        ", int(s0["channel_valid_mask"].sum().item()), "/ 64")

        print("dataset size:", len(dataset))

    # ============================================================
    # 1. 根据配置确定参与实验的数据量
    # ============================================================
    n_total = len(dataset)

    sample_ratio = float(cfg.data.data_sample_ratio)
    sample_seed = int(cfg.data.data_sample_seed)

    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError(
            f"data_sample_ratio must be in (0, 1], got {sample_ratio}"
        )

    n_selected = max(2, int(n_total * sample_ratio))
    n_selected = min(n_selected, n_total)

    # 使用独立的 Python RNG，确保所有 DDP rank
    # 得到完全相同的数据子集，不受 rank-specific seed 影响
    rng = random.Random(sample_seed)

    if n_selected == n_total:
        # 全量训练，不必再做一次随机抽样
        selected_indices = list(range(n_total))
    else:
        # 从全体样本索引中无放回随机抽样
        selected_indices = rng.sample(range(n_total), n_selected)

    # ============================================================
    # 2. 在抽样后的数据子集上划分训练集和验证集
    # ============================================================
    rng.shuffle(selected_indices)

    n_train = int(n_selected * 0.8)
    n_train = max(1, min(n_train, n_selected - 1))

    train_indices = selected_indices[:n_train]
    val_indices = selected_indices[n_train:]

    # ============================================================
    # 3. 日志
    # ============================================================
    if is_main_process():
        print("\n===== Data Subsampling =====")
        print(f"Original samples: {n_total:,}")
        print(f"Sample ratio:     {sample_ratio:.2%}")
        print(f"Selected samples: {n_selected:,}")
        print(f"Train samples:    {len(train_indices):,}")
        print(f"Val samples:      {len(val_indices):,}")
        print(f"Sampling seed:    {sample_seed}")

    train_set = BatchedSubset(dataset, train_indices)
    val_set = BatchedSubset(dataset, val_indices)

    if is_main_process():
        print(f"\n===== Dataset Split =====")
        print(f"train: {len(train_set)}, val: {len(val_set)}")

    num_workers = int(getattr(cfg.train, "num_workers", 2))
    pin_memory = bool(getattr(cfg.train, "pin_memory", True))
    persistent_workers = bool(getattr(cfg.train, "persistent_workers", True))
    prefetch_factor = int(getattr(cfg.train, "prefetch_factor", 4))

    train_batch_sampler = PartAwareDistributedBatchSampler(
        dataset=dataset,
        subset_indices=train_indices,
        batch_size=cfg.train.batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
        seed=cfg.train.seed,
        locality_window_batches=int(getattr(cfg.train, "locality_window_batches", 16)),
        shuffle_within_window=bool(getattr(cfg.train, "shuffle_within_window", True)),
    )

    val_batch_sampler = PartAwareDistributedBatchSampler(
        dataset=dataset,
        subset_indices=val_indices,
        batch_size=cfg.train.val_batch_size if hasattr(cfg.train, "val_batch_size") else cfg.train.batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
        seed=cfg.train.seed,
        locality_window_batches=32,
        shuffle_within_window=False,
    )

    train_loader_kwargs = dict(
        dataset=train_set,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader_kwargs = dict(
        dataset=val_set,
        batch_sampler=val_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    if num_workers > 0:
        train_loader_kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
        )

        val_loader_kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
        )

        # 如果你的 PyTorch 版本支持 in_order=False，可以打开。
        # 远程 IO 场景下，先完成的 worker 可以先返回，可能更快。
        try:
            if "in_order" in DataLoader.__init__.__code__.co_varnames:
                train_loader_kwargs["in_order"] = False
                val_loader_kwargs["in_order"] = False
        except Exception:
            pass

    base_train_loader = DataLoader(**train_loader_kwargs)
    base_val_loader = DataLoader(**val_loader_kwargs)

    use_prefetch_loader = bool(getattr(cfg.train, "use_prefetch_loader", True))
    prefetch_loader_depth = int(getattr(cfg.train, "prefetch_loader_depth", 2))

    if use_prefetch_loader:
        train_loader = PrefetchLoader(
            base_train_loader,
            device=device,
            prefetch=prefetch_loader_depth,
            pin_memory=False,
        )

        val_loader = PrefetchLoader(
            base_val_loader,
            device=device,
            prefetch=prefetch_loader_depth,
            pin_memory=False,
        )
    else:
        train_loader = base_train_loader
        val_loader = base_val_loader

    print("\n===== Building Model =====")
    model = EEGPretrainModel(
        input_dim=patch_len,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        depth=cfg.model.depth,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        n_channels=cfg.data.n_channels,
        n_time_patches=num_time_patches,
        n_bands=n_bands,
        frames_per_patch=frames_per_patch,
    ).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    total_params = count_parameters(model)

    if is_main_process():
        print(' ')
        print("--------------------------------------------------------------------------")
        print(f"Total trainable parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        print("--------------------------------------------------------------------------")
        print(' ')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    criterion = BalancedBandTokenTrajectoryLoss()

    if is_main_process():
        print("\n===== Start Training =====")
    best_val = float("inf")

    if is_main_process():
        ensure_dir_for_file(cfg.train.save_path)

    # print(f"[RANK {rank}] before pre-train barrier", flush=True)
    barrier()
    # print(f"[RANK {rank}] after pre-train barrier", flush=True)

    for epoch in range(cfg.train.num_epochs):
        # print(f"[RANK {rank}] enter epoch {epoch}", flush=True)

        if hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n===== Epoch {epoch + 1}/{cfg.train.num_epochs} =====")

        # print(f"[RANK {rank}] before train_one_epoch epoch={epoch}", flush=True)


        base_dir = os.path.dirname(cfg.train.save_path)
        save_path = os.path.join(base_dir, f"test_checkpoint.pt")

        model_to_save = model.module if distributed else model

        torch.save(
            {
                "model_state_dict": model_to_save.state_dict(),
                "config": cfg,
                "shape_params": shape_params,
            },
            save_path,
        )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cfg=cfg,
            current_epoch=epoch,
        )

        # print(f"[RANK {rank}] after train_one_epoch epoch={epoch}", flush=True)

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        if is_main_process():
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            base_dir = os.path.dirname(cfg.train.save_path)
            base_name = os.path.basename(cfg.train.save_path).replace(".pt", "")

            save_path = os.path.join(base_dir, f"{base_name}_{epoch}.pt")

            model_to_save = model.module if distributed else model

            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "config": cfg,
                    "shape_params": shape_params,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                save_path,
            )

            print(f"Saved model to {save_path}")

            if val_loss < best_val:
                best_val = val_loss

    barrier()

    # if is_main_process():
    #     print("\n===== Visualizing Train Reconstruction =====")
    #
        # model_for_vis = model.module if distributed else model
        #
        # visualize_train_reconstruction(
        #     model=model_for_vis,
        #     loader=train_loader,
        #     cfg=cfg,
        #     device=device,
        #     save_dir=cfg.train.vis_dir,
        #     max_samples=4,
        # )
        #
        # print("\nDone.")
        # print(f"Best val loss: {best_val:.6f}")
        # print(f"Visualization saved to: {cfg.train.vis_dir}")


if __name__ == "__main__":

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    try:
        main()
    finally:
        cleanup_ddp()