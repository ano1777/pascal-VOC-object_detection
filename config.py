from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import numpy as np
import torch

CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

LAYOUT_CLASSES = ('__background__', 'head', 'hand', 'foot')


def classes_for(cfg):
    if cfg.task == 'objects':
        return CLASSES
    if cfg.task == 'layout':
        return LAYOUT_CLASSES
    raise ValueError(f'Unknown task: {cfg.task}')


@dataclass
class Config:
    task: str = "objects"
    crop_padding: float = 0.35
    seed: int = 42
    batch_size: int = 2
    workers: int = 0  # Works in Jupyter, Colab and on Windows.
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    epochs: int = 6
    min_size: int = 480
    max_size: int = 800
    flip_probability: float = 0.5
    freeze_backbone: bool = False
    amp: bool = True
    patience: int = 3
    # None uses the entire split. Small limits are for smoke runs only.
    train_limit: int | None = None
    tune_limit: int | None = None
    eval_limit: int | None = None

    def dictionary(self):
        return asdict(self)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some GPU kernels are nondeterministic; do not promise bitwise reproducibility.


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))
