from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .base import BaseBenchmark
from .base import ToDeviceDataloader
from .stream_prefetch_dataloader import StreamPrefetchDataloader

device  = 'cuda'
batch   = 384
workers = 8

def build_dataset(name: str,
                  split: str,
                  data_root: str = "~/data",
                  *,
                  img_size: int = 224,
                  download: bool = True,
                  **kwargs):
    """
    Returns a torchvision‑style dataset matching `(image, target)` for
    classification or `(image, target_dict)` for detection.
    """
    root = Path(data_root).expanduser()

    if name.lower() == "cifar10":
        tfms = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize((0.5,)*3, (0.5,)*3),
        ])
        train = (split == "train")
        return torchvision.datasets.CIFAR10(root, train=train,
                                transform=tfms,
                                target_transform=None,
                                download=download)

    elif name.lower() == "pets":
        tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        return torchvision.datasets.OxfordIIITPet(root,
                                      split=split,
                                      target_types="category",
                                      transform=tfms,
                                      download=download)

    elif name.lower() == "flowers":
        tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        return torchvision.datasets.Flowers102(root,
                                   split=split,
                                   transform=tfms,
                                   download=download)

    elif name.lower() == "imagenette":
        # https://github.com/fastai/imagenette
        # user downloads .tgz manually or via script, then ImageFolder
        path = root / "imagenette2-160"
        assert path.exists(), "Download Imagenette and untar to data_root first"
        return torchvision.datasets.ImageFolder(path / split,
                                    transform=T.Compose([
                                          T.Resize(img_size),
                                          T.CenterCrop(img_size),
                                          T.ToTensor() ]))

    elif name.lower() == "coco":
        # You must `pip install pycocotools`
        ann_file = (root /
                    f"annotations/instances_{split}2017.json")
        img_dir  = root / f"{split}2017"
        return torchvision.datasets.CocoDetection(img_dir,
                                      ann_file,
                                      transform=T.ToTensor())
    # fallback: still allow FakeData for unit tests
    elif name.lower() == "fake":
        return torchvision.FakeData(size=kwargs.get("n_samples", 10_000),
                                 image_size=(3, img_size, img_size),
                                 num_classes=kwargs.get("n_classes", 10),
                                 transform=T.ToTensor())
    else:
        raise ValueError(f"Unknown dataset {name}")


# 1‑A) DATA
dataset = build_dataset("cifar10", "train", download=True)
loader = DataLoader(dataset, batch_size=batch,
                    num_workers=workers, pin_memory=True)



# 1‑B) MODEL + FWD/BWD
model = torchvision.models.resnet50().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

resnet_benchmark = BaseBenchmark(model, ToDeviceDataloader(loader), optimizer, criterion, 'resnet50')
resnet_prefetch_benchmark = BaseBenchmark(model, StreamPrefetchDataloader(loader), optimizer, criterion, 'resnet50_prefetch')
