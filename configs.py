from dataclasses import dataclass, field


@dataclass
class DataConfig:

    use_multi_lance_parts: bool = True

    # 从全部 Lance 样本中抽取的比例，范围为 (0, 1]
    data_sample_ratio: float = 0.8

    # 数据子集抽样的固定随机种子
    data_sample_seed: int = 42


    set_paths: list = field(default_factory=lambda: [
        "data/sub01_01_EC.set",
    ])

    lance_path: str = "/data/pretrain_processed.lance"

    # 多个 Lance 分片所在的上级目录
    lance_parts_root: str = "s3://omni-mnt/home/jianyu/AAAI2026/processed_parts/"

    # 匹配哪些 Lance 分片目录
    lance_part_glob: str = "part_*.lance"

    storage_options_path: str = "./storage_options.json"



    convert_v_to_uv: bool = True
    clip_seconds: float = 10.0
    clip_stride_seconds: float = 10.0
    target_sfreq: float = 100.0
    n_channels: int = 64
    patch_seconds: float = 1.0
    patch_stride_seconds: float = 1.0
    target_frame_seconds: float = 0.2
    stft_window_seconds: float = 1.0
    stft_hop_seconds: float = 0.1
    band_defs: list = field(default_factory=lambda: [
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma_low", 30.0, 45.0),
    ])
    eps: float = 1e-6






@dataclass
class MaskConfig:
    """
    mask 相关配置。
    """
    mask_ratio: float = 0.9
    min_block_tokens: int = 4
    max_block_tokens: int = 32


@dataclass
class ModelConfig:
    """
    模型相关配置。默认50M的d_model是512
    """
    d_model: int = 512
    n_heads: int = 8
    depth: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"

    # 注意：DDP 下这个是每张 GPU 的 batch size，不是全局 batch size
    batch_size: int = 256

    # Lance 数据读取如果稳定，可以逐步调大：2, 4, 8
    num_workers: int = 4

    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 1

    use_prefetch_loader: bool = False
    prefetch: int = 1

    num_epochs: int = 5

    lr: float = 1e-4
    weight_decay: float = 1e-5

    print_every: int = 5

    # 可选，先关掉，等 DDP 跑通后再开
    use_amp: bool = False

    output_root: str = "/tos/mnt/technicalReport/50M_maskRatio_90_1Param_dataRatio_08"
    save_path: str = "/tos/mnt/technicalReport/50M_maskRatio_90_1Param_dataRatio_08"
    vis_dir: str = "/tos/mnt/technicalReport/50M_maskRatio_90_1Param_dataRatio_08"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
