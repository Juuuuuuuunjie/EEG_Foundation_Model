import torch
import numpy as np
import torch.distributed as dist
import time

from masking import generate_block_mask, apply_token_mask


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def reduce_loss_sum_and_count(loss_sum, count, device):
    """
    汇总所有 rank 的 loss_sum 和 count，返回全局平均 loss。
    """
    if not is_dist_avail_and_initialized():
        return loss_sum / max(1, count)

    t = torch.tensor(
        [loss_sum, count],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    global_loss_sum = t[0].item()
    global_count = int(t[1].item())

    return global_loss_sum / max(1, global_count)


def build_batch_token_mask(batch_size, num_tokens, cfg, device):
    masks = np.stack([
        generate_block_mask(
            num_tokens=num_tokens,
            mask_ratio=cfg.mask.mask_ratio,
        )
        for _ in range(batch_size)
    ], axis=0)

    return torch.from_numpy(masks).float().to(device, non_blocking=True)


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


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    cfg,
    current_epoch
):
    model.train()

    rank = get_rank()
    print(f"[TRAIN] rank={rank} enter train_one_epoch", flush=True)

    # end = time.time()

    iterator = iter(loader)
    print(f"[TRAIN] rank={rank} iterator created", flush=True)



    total_loss = 0.0
    total_steps = 0

    use_amp = getattr(cfg.train, "use_amp", False)
    scaler = getattr(train_one_epoch, "_scaler", None)

    if use_amp and scaler is None:
        scaler = torch.cuda.amp.GradScaler()
        train_one_epoch._scaler = scaler

    for step, batch in enumerate(loader):

        # data_time = time.time() - end
        # compute_start = time.time()

        batch = move_batch_to_device(batch, device)

        token_inputs = batch["token_inputs"]                    # [B, S, L]
        targets = batch["targets"]                              # [B, S, F, K]
        token_channel_indices = batch["token_channel_indices"]  # [B, S]
        token_time_indices = batch["token_time_indices"]        # [B, S]
        token_valid_mask = batch["token_valid_mask"]            # [B, S]

        B, S, _ = token_inputs.shape

        token_mask = build_batch_token_mask(
            batch_size=B,
            num_tokens=S,
            cfg=cfg,
            device=device,
        )

        masked_inputs = apply_token_mask(
            tokens=token_inputs,
            token_mask=token_mask,
        )

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                pred = model(
                    token_inputs=masked_inputs,
                    token_channel_indices=token_channel_indices,
                    token_time_indices=token_time_indices,
                )

                loss = criterion(
                    pred=pred,
                    target=targets,
                    token_valid_mask=token_valid_mask,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            pred = model(
                token_inputs=masked_inputs,
                token_channel_indices=token_channel_indices,
                token_time_indices=token_time_indices,
            )

            loss = criterion(
                pred=pred,
                target=targets,
                token_valid_mask=token_valid_mask,
            )

            loss.backward()
            optimizer.step()

        torch.cuda.synchronize(device)
        # compute_time = time.time() - compute_start

        loss_value = loss.detach().item()
        total_loss += loss_value
        total_steps += 1

        if is_main_process() and step % cfg.train.print_every == 0:
            print(
                f"Epoch={current_epoch}, "
                f"Step={step}, "
                f"Loss={loss_value:.6f}, "
                f"input_mask_ratio={token_mask.mean().item():.3f}",
                flush=True,
            )

            # print(
            #     f"[TRAIN TIME] rank={rank} step={step} "
            #     f"data_time={data_time:.3f}s "
            #     f"compute_time={compute_time:.3f}s "
            #     f"loss={loss_value:.6f}",
            #     flush=True,
            # )

    avg_loss = reduce_loss_sum_and_count(
        loss_sum=total_loss,
        count=total_steps,
        device=device,
    )

    return avg_loss


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    cfg,
):
    model.eval()

    total_loss = 0.0
    total_steps = 0

    use_amp = getattr(cfg.train, "use_amp", False)

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        token_inputs = batch["token_inputs"]
        targets = batch["targets"]
        token_channel_indices = batch["token_channel_indices"]
        token_time_indices = batch["token_time_indices"]
        token_valid_mask = batch["token_valid_mask"]

        B, S, _ = token_inputs.shape

        token_mask = build_batch_token_mask(
            batch_size=B,
            num_tokens=S,
            cfg=cfg,
            device=device,
        )

        masked_inputs = apply_token_mask(
            tokens=token_inputs,
            token_mask=token_mask,
        )

        if use_amp:
            with torch.cuda.amp.autocast():
                pred = model(
                    token_inputs=masked_inputs,
                    token_channel_indices=token_channel_indices,
                    token_time_indices=token_time_indices,
                )

                loss = criterion(
                    pred=pred,
                    target=targets,
                    token_valid_mask=token_valid_mask,
                )
        else:
            pred = model(
                token_inputs=masked_inputs,
                token_channel_indices=token_channel_indices,
                token_time_indices=token_time_indices,
            )

            loss = criterion(
                pred=pred,
                target=targets,
                token_valid_mask=token_valid_mask,
            )

        total_loss += loss.detach().item()
        total_steps += 1

    avg_loss = reduce_loss_sum_and_count(
        loss_sum=total_loss,
        count=total_steps,
        device=device,
    )

    return avg_loss
