import queue
import threading

import torch


def _pin_memory_batch(batch):
    if torch.is_tensor(batch):
        if batch.is_cuda:
            return batch

        try:
            if batch.is_pinned():
                return batch
        except RuntimeError:
            pass

        return batch.pin_memory()

    if isinstance(batch, dict):
        return {
            k: _pin_memory_batch(v)
            for k, v in batch.items()
        }

    if isinstance(batch, list):
        return [
            _pin_memory_batch(v)
            for v in batch
        ]

    if isinstance(batch, tuple):
        return tuple(
            _pin_memory_batch(v)
            for v in batch
        )

    return batch


def _move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)

    if isinstance(batch, dict):
        return {
            k: _move_batch_to_device(v, device)
            for k, v in batch.items()
        }

    if isinstance(batch, list):
        return [
            _move_batch_to_device(v, device)
            for v in batch
        ]

    if isinstance(batch, tuple):
        return tuple(
            _move_batch_to_device(v, device)
            for v in batch
        )

    return batch


def _record_stream_batch(batch, stream):
    if torch.is_tensor(batch):
        if batch.is_cuda:
            batch.record_stream(stream)
        return

    if isinstance(batch, dict):
        for v in batch.values():
            _record_stream_batch(v, stream)
        return

    if isinstance(batch, list):
        for v in batch:
            _record_stream_batch(v, stream)
        return

    if isinstance(batch, tuple):
        for v in batch:
            _record_stream_batch(v, stream)
        return


class PrefetchLoader:
    """
    DataLoader 外层 CUDA prefetch。

    职责：
        1. 后台线程提前从 DataLoader 拉 CPU batch；
        2. 可选 pin_memory；
        3. 用独立 CUDA stream 提前把 batch 搬到 GPU；
        4. 当前 batch 计算时，下一个 batch 正在 H2D。

    注意：
        它主要隐藏 CPU->GPU 拷贝。
        如果远程 S3/Lance 读取本身要 5 秒，而 GPU 计算 1 秒，
        PrefetchLoader 只能隐藏其中一部分，根本优化仍然要靠：
            - PartAwareDistributedBatchSampler
            - MultiLanceEEGPretrainDataset.__getitems__ batch take
            - DataLoader num_workers/prefetch_factor
    """

    def __init__(
        self,
        loader,
        device,
        prefetch=2,
        pin_memory=False,
    ):
        self.loader = loader
        self.device = torch.device(device)
        self.prefetch = max(1, int(prefetch))
        self.pin_memory = bool(pin_memory)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        q = queue.Queue(maxsize=self.prefetch)
        stop_token = object()
        error_holder = []

        def producer():
            try:
                for batch in self.loader:
                    if self.pin_memory:
                        batch = _pin_memory_batch(batch)
                    q.put(batch)
            except BaseException as exc:
                error_holder.append(exc)
            finally:
                q.put(stop_token)

        thread = threading.Thread(
            target=producer,
            daemon=True,
        )
        thread.start()

        transfer_stream = torch.cuda.Stream(device=self.device)

        def preload():
            batch = q.get()

            if batch is stop_token:
                return None

            if error_holder:
                raise error_holder[0]

            with torch.cuda.stream(transfer_stream):
                batch = _move_batch_to_device(batch, self.device)

            return batch

        next_batch = preload()

        while next_batch is not None:
            current_stream = torch.cuda.current_stream(device=self.device)

            # 等待 H2D stream 完成当前 next_batch 的搬运
            current_stream.wait_stream(transfer_stream)

            current_batch = next_batch

            # 告诉 CUDA allocator：这些 tensor 后续会在 current_stream 使用
            _record_stream_batch(current_batch, current_stream)

            if error_holder:
                raise error_holder[0]

            # 开始预取下一个 batch。
            # 这一步会在 transfer_stream 上排队 H2D，
            # 然后 yield current_batch 让训练计算和下一批搬运重叠。
            next_batch = preload()

            yield current_batch

        if error_holder:
            raise error_holder[0]
