import bisect
import glob
import json
import math
import os
import random
import re
from urllib.parse import urlparse

import lance
import numpy as np
import pyarrow.fs as pafs
import torch
from torch.utils.data import Dataset, Sampler


def _normalize_s3_endpoint_for_pyarrow(endpoint, bucket=None):
    """
    把 storage_options 里的 endpoint 转成 pyarrow.fs.S3FileSystem 可用的 endpoint_override。

    例如：
        endpoint = "https://jianyu.tos-s3-cn-beijing.ivolces.com"
        bucket   = "jianyu"

    返回：
        "tos-s3-cn-beijing.ivolces.com"

    原因：
        pyarrow 的 endpoint_override 应该是 S3 服务 endpoint，
        bucket 仍然从 s3://bucket/key 里面取。
    """
    if not endpoint:
        return None, "https"

    endpoint = str(endpoint).strip().rstrip("/")

    if "://" not in endpoint:
        endpoint_for_parse = "https://" + endpoint
    else:
        endpoint_for_parse = endpoint

    parsed = urlparse(endpoint_for_parse)

    scheme = parsed.scheme or "https"
    host = parsed.netloc or parsed.path
    host = host.strip().rstrip("/")

    # 如果 endpoint 写成了 bucket.endpoint.com，则去掉 bucket.
    if bucket:
        bucket_prefix = bucket + "."
        if host.startswith(bucket_prefix):
            host = host[len(bucket_prefix):]

    return host, scheme


def load_storage_options(storage_options_path):
    with open(storage_options_path, "r", encoding="utf-8") as f:
        storage_options = json.load(f)
    return {k: str(v) for k, v in storage_options.items()}


def _normalize_lance_path(path):
    path = str(path)

    if path.endswith(".lance"):
        path = path + "/"

    return path


def _lance_part_sort_key(path):
    name = path.rstrip("/").split("/")[-1]
    m = re.search(r"part_(\d+)_(\d+)\.lance$", name)

    if m is None:
        return (float("inf"), float("inf"), name)

    return (int(m.group(1)), int(m.group(2)), name)


def _build_pyarrow_s3_options(storage_options=None, bucket=None):
    """
    把 storage_options.json 转成 pyarrow.fs.S3FileSystem 参数。

    支持火山 TOS S3 endpoint，例如：
        "aws_endpoint": "https://jianyu.tos-s3-cn-beijing.ivolces.com"
    """
    if storage_options is None:
        storage_options = {}

    opts = dict(storage_options)

    access_key = (
        opts.get("access_key_id")
        or opts.get("aws_access_key_id")
        or opts.get("AWS_ACCESS_KEY_ID")
        or opts.get("key")
        or opts.get("access_key")
    )

    secret_key = (
        opts.get("secret_access_key")
        or opts.get("aws_secret_access_key")
        or opts.get("AWS_SECRET_ACCESS_KEY")
        or opts.get("secret")
        or opts.get("secret_key")
    )

    session_token = (
        opts.get("session_token")
        or opts.get("aws_session_token")
        or opts.get("AWS_SESSION_TOKEN")
        or opts.get("token")
    )

    endpoint = (
        opts.get("aws_endpoint")
        or opts.get("endpoint")
        or opts.get("endpoint_url")
        or opts.get("aws_endpoint_url")
        or opts.get("AWS_ENDPOINT")
        or opts.get("AWS_ENDPOINT_URL")
    )

    region = (
        opts.get("region")
        or opts.get("aws_region")
        or opts.get("AWS_REGION")
        or opts.get("AWS_DEFAULT_REGION")
        or "cn-beijing"
    )

    endpoint_override, scheme = _normalize_s3_endpoint_for_pyarrow(
        endpoint=endpoint,
        bucket=bucket,
    )

    kwargs = {
        "region": region,
    }

    if access_key:
        kwargs["access_key"] = access_key

    if secret_key:
        kwargs["secret_key"] = secret_key

    if session_token:
        kwargs["session_token"] = session_token

    if endpoint_override:
        kwargs["endpoint_override"] = endpoint_override
        kwargs["scheme"] = scheme

        # 火山 TOS 常见 bucket endpoint 形式。
        kwargs["force_virtual_addressing"] = True

    return kwargs


def discover_lance_part_paths(
    root_path,
    storage_options=None,
    glob_pattern="part_*.lance",
):
    """
    自动扫描 Lance 分片目录。

    支持：
        本地目录
        s3://bucket/path/
    """
    root_path = str(root_path).rstrip("/")

    # 本地路径
    if not root_path.startswith("s3://"):
        search_pattern = os.path.join(root_path, glob_pattern)
        paths = glob.glob(search_pattern)

        part_paths = [
            _normalize_lance_path(os.path.abspath(p))
            for p in paths
        ]

        part_paths = sorted(part_paths, key=_lance_part_sort_key)

        if len(part_paths) == 0:
            raise FileNotFoundError(
                f"No Lance parts found under root_path={root_path}, "
                f"glob_pattern={glob_pattern}"
            )

        return part_paths

    # S3 路径
    s3_path = root_path[len("s3://"):]
    bucket, _, prefix = s3_path.partition("/")

    bucket = bucket.strip("/")
    prefix = prefix.strip("/")

    s3_kwargs = _build_pyarrow_s3_options(
        storage_options=storage_options,
        bucket=bucket,
    )

    safe_kwargs = {
        k: v for k, v in s3_kwargs.items()
        if k not in ("access_key", "secret_key", "session_token")
    }

    print("[S3 list] root_path:", root_path, flush=True)
    print("[S3 list] bucket:", bucket, flush=True)
    print("[S3 list] prefix:", prefix, flush=True)
    print("[S3 list] pyarrow s3 kwargs:", safe_kwargs, flush=True)

    if "endpoint_override" not in s3_kwargs:
        raise ValueError(
            "storage_options does not contain aws_endpoint/endpoint/AWS_ENDPOINT_URL. "
            "Without endpoint_override, pyarrow will try default AWS endpoint."
        )

    try:
        fs = pafs.S3FileSystem(**s3_kwargs)

        print(fs, flush=True)
        print(s3_kwargs, flush=True)

    except TypeError as e:
        if "force_virtual_addressing" in str(e) or "unexpected keyword" in str(e):
            print(
                "[S3 list] current pyarrow does not support force_virtual_addressing, "
                "retry without it",
                flush=True,
            )
            s3_kwargs = dict(s3_kwargs)
            s3_kwargs.pop("force_virtual_addressing", None)
            fs = pafs.S3FileSystem(**s3_kwargs)
        else:
            raise

    base_dir = f"{bucket}/{prefix}" if prefix else bucket

    print("[S3 list] base_dir:", base_dir, flush=True)

    selector = pafs.FileSelector(
        base_dir=base_dir,
        recursive=False,
    )

    infos = fs.get_file_info(selector)

    pattern_re = re.compile(
        "^" + re.escape(glob_pattern).replace("\\*", ".*") + "$"
    )

    part_paths = []

    for info in infos:
        name = info.base_name

        if not pattern_re.match(name):
            continue

        full_path = "s3://" + info.path
        part_paths.append(_normalize_lance_path(full_path))

    part_paths = sorted(part_paths, key=_lance_part_sort_key)

    if len(part_paths) == 0:
        raise FileNotFoundError(
            f"No Lance parts found under root_path={root_path}, "
            f"glob_pattern={glob_pattern}, "
            f"base_dir={base_dir}"
        )

    print(f"[S3 list] found {len(part_paths)} lance parts", flush=True)
    print("[S3 list] first part:", part_paths[0], flush=True)
    print("[S3 list] last part:", part_paths[-1], flush=True)

    return part_paths


class LanceEEGPretrainDataset(Dataset):
    """
    单个 .lance 分片数据集。

    优化点：
    1. worker 内懒加载 lance.dataset；
    2. __getstate__ 时清空 ds，兼容 spawn；
    3. __getitems__ 内部排序 indices，批量 take，之后恢复原顺序；
    4. 不要逐条 ds.take([idx])。
    """

    COLUMNS = [
        "num_tokens",
        "patch_len",
        "n_bands",
        "frames_per_patch",
        "n_channels",
        "token_inputs",
        "targets",
        "token_channel_indices",
        "token_time_indices",
        "token_valid_mask",
        "channel_valid_mask",
    ]

    def __init__(self, lance_path, storage_options=None, length=None):
        self.lance_path = _normalize_lance_path(lance_path)
        self.storage_options = storage_options
        self.ds = None

        if length is None:
            self.length = self._get_length_once()
        else:
            self.length = int(length)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ds"] = None
        return state

    def reset_worker_state(self):
        self.ds = None

    def _open(self):
        if self.ds is None:
            if self.storage_options is None:
                self.ds = lance.dataset(self.lance_path)
            else:
                self.ds = lance.dataset(
                    self.lance_path,
                    storage_options=self.storage_options,
                )
        return self.ds

    def _get_length_once(self):
        if self.storage_options is None:
            ds = lance.dataset(self.lance_path)
        else:
            ds = lance.dataset(
                self.lance_path,
                storage_options=self.storage_options,
            )

        n = ds.count_rows()
        del ds
        return int(n)

    def __len__(self):
        return self.length

    @staticmethod
    def _row_to_sample(row):
        num_tokens = int(row["num_tokens"])
        patch_len = int(row["patch_len"])
        n_bands = int(row["n_bands"])
        frames_per_patch = int(row["frames_per_patch"])
        n_channels = int(row["n_channels"])

        token_inputs = np.asarray(row["token_inputs"], dtype=np.float32).reshape(
            num_tokens, patch_len
        )

        targets = np.asarray(row["targets"], dtype=np.float32).reshape(
            num_tokens, n_bands, frames_per_patch
        )

        token_channel_indices = np.asarray(
            row["token_channel_indices"],
            dtype=np.int64,
        )

        token_time_indices = np.asarray(
            row["token_time_indices"],
            dtype=np.int64,
        )

        token_valid_mask = np.asarray(
            row["token_valid_mask"],
            dtype=np.float32,
        )

        channel_valid_mask = np.asarray(
            row["channel_valid_mask"],
            dtype=np.float32,
        )

        if channel_valid_mask.shape[0] != n_channels:
            raise ValueError(
                f"channel_valid_mask shape mismatch: "
                f"{channel_valid_mask.shape[0]} != {n_channels}"
            )

        return {
            "token_inputs": torch.from_numpy(token_inputs),
            "targets": torch.from_numpy(targets),
            "token_channel_indices": torch.from_numpy(token_channel_indices),
            "token_time_indices": torch.from_numpy(token_time_indices),
            "token_valid_mask": torch.from_numpy(token_valid_mask),
            "channel_valid_mask": torch.from_numpy(channel_valid_mask),
        }

    @classmethod
    def _table_to_samples(cls, table):
        rows = table.to_pylist()
        return [cls._row_to_sample(row) for row in rows]

    def __getitem__(self, idx):
        idx = int(idx)
        ds = self._open()

        table = ds.take(
            indices=[idx],
            columns=self.COLUMNS,
        )

        row = table.to_pylist()[0]
        return self._row_to_sample(row)

    def __getitems__(self, indices):
        if len(indices) == 0:
            return []

        indices = [int(i) for i in indices]

        # 为了远程 IO 局部性，先按 row index 排序读取
        sorted_pairs = sorted(
            enumerate(indices),
            key=lambda x: x[1],
        )

        sorted_out_positions = [x[0] for x in sorted_pairs]
        sorted_indices = [x[1] for x in sorted_pairs]

        ds = self._open()

        table = ds.take(
            indices=sorted_indices,
            columns=self.COLUMNS,
        )

        sorted_items = self._table_to_samples(table)

        # 恢复 DataLoader 请求的原顺序
        results = [None] * len(indices)

        for out_pos, item in zip(sorted_out_positions, sorted_items):
            results[out_pos] = item

        return results


class MultiLanceEEGPretrainDataset(Dataset):
    """
    多个 .lance 分片包装成一个总数据集。

    这个版本是给远程 S3/TOS 训练优化过的：

    1. 不在 __init__ 长期保存 lance.dataset 对象；
    2. worker 内 lazy open，并缓存到 self._datasets；
    3. __getitems__ 按 part 分组；
    4. 每个 part 内按 local_idx 排序；
    5. 每个 part 只调用一次 ds.take(indices=[...])；
    6. 支持 _locate / locate，方便 PartAwareDistributedBatchSampler 使用；
    7. 支持 spawn，多 worker 时不会 pickle lance dataset handle。
    """

    COLUMNS = LanceEEGPretrainDataset.COLUMNS

    def __init__(self, lance_paths, storage_options=None, lengths=None):
        if len(lance_paths) == 0:
            raise ValueError("lance_paths is empty")

        self.lance_paths = [
            _normalize_lance_path(p)
            for p in lance_paths
        ]

        self.storage_options = storage_options

        # worker 内懒加载的 lance dataset 句柄
        self._datasets = {}

        if lengths is None:
            self.lengths = self._count_all_lengths()
        else:
            if len(lengths) != len(self.lance_paths):
                raise ValueError(
                    f"lengths size mismatch: {len(lengths)} != {len(self.lance_paths)}"
                )
            self.lengths = [int(x) for x in lengths]

        self.cumulative_sizes = []

        total = 0
        for n in self.lengths:
            total += int(n)
            self.cumulative_sizes.append(total)

        if len(self.cumulative_sizes) == 0 or self.cumulative_sizes[-1] <= 0:
            raise ValueError("total dataset length is zero")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_datasets"] = {}
        return state

    def reset_worker_state(self):
        self._datasets = {}

    def _open_lance_path(self, path):
        if self.storage_options is None:
            return lance.dataset(path)

        return lance.dataset(
            path,
            storage_options=self.storage_options,
        )

    def _count_all_lengths(self):
        lengths = []

        print(
            f"[MultiLance] counting rows for {len(self.lance_paths)} lance parts...",
            flush=True,
        )

        for i, path in enumerate(self.lance_paths):
            ds = self._open_lance_path(path)
            n = int(ds.count_rows())
            lengths.append(n)
            del ds

            if (i + 1) % 50 == 0 or i == 0 or i + 1 == len(self.lance_paths):
                print(
                    f"[MultiLance] counted {i + 1}/{len(self.lance_paths)} parts, "
                    f"current rows={n}",
                    flush=True,
                )

        print(
            f"[MultiLance] total rows={sum(lengths)}",
            flush=True,
        )

        return lengths

    def __len__(self):
        return int(self.cumulative_sizes[-1])

    @property
    def num_parts(self):
        return len(self.lance_paths)

    def _get_dataset(self, part_id):
        part_id = int(part_id)

        if part_id < 0 or part_id >= len(self.lance_paths):
            raise IndexError(f"part_id {part_id} out of range")

        if part_id not in self._datasets:
            self._datasets[part_id] = self._open_lance_path(
                self.lance_paths[part_id]
            )

        return self._datasets[part_id]

    def _locate_index(self, idx):
        idx = int(idx)

        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for dataset length {len(self)}")

        part_id = bisect.bisect_right(self.cumulative_sizes, idx)
        prev_cum = 0 if part_id == 0 else self.cumulative_sizes[part_id - 1]
        local_idx = idx - prev_cum

        return int(part_id), int(local_idx)

    # 给 sampler 用
    def _locate(self, idx):
        return self._locate_index(idx)

    # 给 sampler 用
    def locate(self, idx):
        return self._locate_index(idx)

    @staticmethod
    def _row_to_sample(row):
        return LanceEEGPretrainDataset._row_to_sample(row)

    @classmethod
    def _table_to_samples(cls, table):
        rows = table.to_pylist()
        return [cls._row_to_sample(row) for row in rows]

    # 兼容你之前代码里写的 _table_to_items 名字
    @classmethod
    def _table_to_items(cls, table):
        return cls._table_to_samples(table)

    def __getitem__(self, idx):
        part_id, local_idx = self._locate_index(int(idx))

        ds = self._get_dataset(part_id)

        table = ds.take(
            indices=[int(local_idx)],
            columns=self.COLUMNS,
        )

        row = table.to_pylist()[0]
        return self._row_to_sample(row)

    def __getitems__(self, indices):
        """
        DataLoader 在 map-style dataset 上可能会调用这个方法。
        这里是真正加速的关键：

        原来慢的方式：
            for idx in indices:
                ds.take([idx])

        现在的方式：
            按 part 分组
            每个 part 内排序
            每个 part 执行一次 ds.take([...])
        """
        if len(indices) == 0:
            return []

        indices = [int(i) for i in indices]

        grouped = {}

        for out_pos, global_idx in enumerate(indices):
            part_id, local_idx = self._locate_index(global_idx)

            grouped.setdefault(part_id, []).append(
                (out_pos, local_idx)
            )

        results = [None] * len(indices)

        for part_id, entries in grouped.items():
            ds = self._get_dataset(part_id)

            # 为了远程 IO 局部性，按 local_idx 排序
            sorted_entries = sorted(
                entries,
                key=lambda x: x[1],
            )

            sorted_out_positions = [x[0] for x in sorted_entries]
            sorted_local_indices = [x[1] for x in sorted_entries]

            table = ds.take(
                indices=sorted_local_indices,
                columns=self.COLUMNS,
            )

            sorted_items = self._table_to_samples(table)

            # 恢复 DataLoader 请求的原顺序
            for out_pos, item in zip(sorted_out_positions, sorted_items):
                results[out_pos] = item

        return results


class PartAwareDistributedBatchSampler(Sampler):
    """
    让每个 batch 尽量来自同一个 Lance part，并且 part 内 row 尽量局部连续。

    这个对 S3/TOS 上的 Lance 数据非常重要。

    普通 DistributedSampler(shuffle=True) 会导致：
        一个 batch 里样本来自很多 part，且 row 完全随机。

    这个 sampler 会尽量做到：
        一个 batch 主要来自同一个 part；
        part 内 local_idx 尽量局部连续；
        DDP 下每个 rank 拿不同 batch；
        每个 epoch 可以 shuffle part/window/batch。
    """

    def __init__(
        self,
        dataset,
        subset_indices,
        batch_size,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=42,
        locality_window_batches=16,
        shuffle_within_window=True,
    ):
        self.dataset = dataset
        self.subset_indices = [int(x) for x in subset_indices]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.locality_window_batches = int(locality_window_batches)
        self.shuffle_within_window = bool(shuffle_within_window)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")

        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank should be in [0, {self.num_replicas}), got {self.rank}"
            )

        self._build_records()

    def _locate(self, global_idx):
        if hasattr(self.dataset, "_locate"):
            return self.dataset._locate(global_idx)

        if hasattr(self.dataset, "locate"):
            return self.dataset.locate(global_idx)

        if hasattr(self.dataset, "_locate_index"):
            return self.dataset._locate_index(global_idx)

        raise AttributeError(
            "dataset must provide _locate(global_idx), locate(global_idx), "
            "or _locate_index(global_idx), returning (part_id, local_idx)"
        )

    def _build_records(self):
        """
        records_by_part[part_id] = [(subset_local_idx, local_idx_in_part), ...]

        注意：
        DataLoader 的 dataset 是 train_set/val_set，即 BatchedSubset。
        所以 batch_sampler 输出的 index 必须是 subset-local index。
        """
        self.records_by_part = {}

        for subset_local_idx, global_idx in enumerate(self.subset_indices):
            part_id, local_idx = self._locate(global_idx)
            part_id = int(part_id)
            local_idx = int(local_idx)

            self.records_by_part.setdefault(part_id, []).append(
                (subset_local_idx, local_idx)
            )

        # 每个 part 内按 local_idx 排序，增强 IO 局部性
        for part_id in self.records_by_part:
            self.records_by_part[part_id].sort(key=lambda x: x[1])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _make_all_batches(self):
        rng = random.Random(self.seed + self.epoch)

        all_batches = []

        part_ids = list(self.records_by_part.keys())

        if self.shuffle:
            rng.shuffle(part_ids)

        for part_id in part_ids:
            records = self.records_by_part[part_id]

            window_size = max(
                self.batch_size,
                self.batch_size * self.locality_window_batches,
            )

            windows = [
                records[i:i + window_size]
                for i in range(0, len(records), window_size)
            ]

            if self.shuffle:
                rng.shuffle(windows)

            for window in windows:
                # 如果你想极限追求 IO 局部性，可以设置 shuffle_within_window=False
                if self.shuffle and self.shuffle_within_window:
                    rng.shuffle(window)

                for i in range(0, len(window), self.batch_size):
                    batch_records = window[i:i + self.batch_size]

                    if len(batch_records) < self.batch_size and self.drop_last:
                        continue

                    batch_subset_indices = [x[0] for x in batch_records]
                    all_batches.append(batch_subset_indices)

        if self.shuffle:
            rng.shuffle(all_batches)

        # DDP 下保证每个 rank batch 数一致
        if self.num_replicas > 1:
            if self.drop_last:
                total_size = (
                    len(all_batches) // self.num_replicas
                ) * self.num_replicas
                all_batches = all_batches[:total_size]
            else:
                total_size = int(
                    math.ceil(len(all_batches) / self.num_replicas)
                ) * self.num_replicas

                if len(all_batches) < total_size:
                    padding_size = total_size - len(all_batches)
                    padding = all_batches[:padding_size]
                    all_batches = all_batches + padding

            all_batches = all_batches[self.rank:len(all_batches):self.num_replicas]

        return all_batches

    def __iter__(self):
        batches = self._make_all_batches()

        for batch in batches:
            yield batch

    def __len__(self):
        total_batches = 0

        for records in self.records_by_part.values():
            if self.drop_last:
                total_batches += len(records) // self.batch_size
            else:
                total_batches += math.ceil(len(records) / self.batch_size)

        if self.num_replicas > 1:
            if self.drop_last:
                return total_batches // self.num_replicas
            return math.ceil(total_batches / self.num_replicas)

        return total_batches
