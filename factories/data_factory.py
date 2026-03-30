
import os
import csv
import random
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from io import BytesIO
import pandas as pd



def _ensure_dir_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[data_factory] Directory not found: {path}")

def _pick_int(d: Dict, key: str, default: int) -> int:
    try:
        return int(d.get(key, default))
    except Exception:
        return default
    
def _load_client_pkl(root: str, cid: int, split: str):

    pkl_path = os.path.join(root, split, f"client{cid}.pkl")
    if not os.path.exists(pkl_path):
        return None

    ds = torch.load(pkl_path, weights_only=False)
    return ds


def _stratified_split_by_labels(labels: List[int], val_ratio: float = 0.2, seed: int = 2025) -> Tuple[List[int], List[int]]:
    """给定每个样本的label，返回 (train_idx, val_idx)，做一个最小分层随机切分。"""
    rng = random.Random(seed)
    by_cls: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        by_cls.setdefault(y, []).append(idx)
    tr_idx, va_idx = [], []
    for _, idxs in by_cls.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, int(round(n * val_ratio))) if n > 4 else max(1, n // 5)
        va_idx.extend(idxs[:n_val])
        tr_idx.extend(idxs[n_val:])
    tr_idx.sort(); va_idx.sort()
    return tr_idx, va_idx



class ImageListDataset(Dataset):

    def __init__(self, root: str, relpaths: List[str], transform=None):
        _ensure_dir_exists(root)
        self.root = root
        self.paths = relpaths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        rp = self.paths[i]
        p = os.path.join(self.root, rp)
        img = Image.open(p).convert("RGB")
        class_name = os.path.basename(os.path.dirname(p))
        if self.transform is not None:
            img = self.transform(img)
        return img, class_name


def _build_label_map_from_samples(root: str, relpaths: List[str]) -> Dict[str, int]:
    """
    从样本路径推断“类别名集合”，类别名 = 倒数第二级目录名。
    """
    classes = set()
    for rp in relpaths:
        p = os.path.join(root, rp)
        cls = os.path.basename(os.path.dirname(p))
        classes.add(cls)
    class_names = sorted(classes)
    if len(class_names) == 0:
        raise RuntimeError("[data_factory] No classes detected from samples.")
    return {c: i for i, c in enumerate(class_names)}



class AmazonTagDataset(Dataset):

    def __init__(self, tags: List[str], text_field: str = "text", label_field: str = "label"):
        self.tags = tags
        self.text_field = text_field
        self.label_field = label_field
        self._cache: Dict[str, List[Dict[str, str]]] = {}

    def _ensure_loaded(self, path: str):
        if path in self._cache:
            return
        with open(path, "r", encoding="utf-8") as f:
            rdr = list(csv.DictReader(f))
        if len(rdr) == 0:
            raise RuntimeError(f"[data_factory] Empty CSV: {path}")
        if self.text_field not in rdr[0] or self.label_field not in rdr[0]:
            raise KeyError(
                f"[data_factory] CSV columns missing: '{self.text_field}' or '{self.label_field}' in {path}. "
                f"Available: {list(rdr[0].keys())}"
            )
        self._cache[path] = rdr

    def __len__(self) -> int:
        return len(self.tags)

    def __getitem__(self, i: int):
        tag = self.tags[i]
        if "#" not in tag:
            raise ValueError(f"[data_factory] Bad tag (expect csv#row_idx): {tag}")
        path, idx_str = tag.split("#", 1)
        idx = int(idx_str)
        self._ensure_loaded(path)
        row = self._cache[path][idx]
        text = row[self.text_field]
        label = int(row[self.label_field])
        return text, label



class DomainNetHFDataset(Dataset):
    def __init__(self, root: str, tags: List[str]):
        _ensure_dir_exists(root)
        self.root = root
        self.tags = tags
        self._cache: Dict[str, pd.DataFrame] = {}  # parquet_rel -> DataFrame(image,label)

    def _ensure_loaded(self, pq_rel: str):
        if pq_rel in self._cache:
            return
        pq_abs = os.path.join(self.root, pq_rel)
        df = pd.read_parquet(pq_abs, columns=["image", "label"])
        self._cache[pq_rel] = df

    def __len__(self) -> int:
        return len(self.tags)

    def __getitem__(self, i: int):
        tag = self.tags[i]
        if "#" not in tag:
            raise ValueError(f"[data_factory] Bad tag (expect parquet_rel#row_idx): {tag}")
        pq_rel, idx_str = tag.split("#", 1)
        idx = int(idx_str)
        self._ensure_loaded(pq_rel)
        row = self._cache[pq_rel].iloc[idx]
        img_field = row["image"]
        if isinstance(img_field, dict):
            if "bytes" in img_field and img_field["bytes"] is not None:
                img = Image.open(BytesIO(img_field["bytes"])).convert("RGB")
            elif "path" in img_field and img_field["path"] is not None:
                img = Image.open(img_field["path"]).convert("RGB")
            else:
                raise ValueError(f"[data_factory] Unsupported image dict format at {pq_rel}#{idx}")
        elif isinstance(img_field, (bytes, bytearray)):
            img = Image.open(BytesIO(img_field)).convert("RGB")
        else:
            p = str(img_field)
            img = Image.open(p).convert("RGB")
        label = int(row["label"])
        return img, label


def _load_labels_for_domainnet_hf(root: str, tags: List[str]) -> List[int]:
    """tags: ['data/train-0000.parquet#123', ...] -> labels"""
    cache: Dict[str, pd.DataFrame] = {}
    labels: List[int] = []
    bucket: Dict[str, List[int]] = {}
    for t in tags:
        pq_rel, idx_str = t.split("#", 1)
        bucket.setdefault(pq_rel, []).append(int(idx_str))
    for pq_rel, ridxs in bucket.items():
        pq_abs = os.path.join(root, pq_rel)
        if pq_rel not in cache:
            df = pd.read_parquet(pq_abs, columns=["label"])
            cache[pq_rel] = df
        df = cache[pq_rel]
        for i in ridxs:
            labels.append(int(df.iloc[i]["label"]))
    return labels



def build_dataloaders_from_spec(
    cfg: Dict,
    spec,
    tok_or_proc
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    batch_size = _pick_int(train_cfg, "batch_size", 8)
    val_ratio = float(data_cfg.get("val_ratio", 0.2))
    max_len = _pick_int(data_cfg, "max_length", 256)

    dataset_name = getattr(spec, "dataset", None)
    if dataset_name is None:
        raise AttributeError("[data_factory] 'spec.dataset' missing.")

    if dataset_name in ["domainnet", "officehome", "tinyimagenet"]:
        root = cfg["dataset"]["root"]
        _ensure_dir_exists(root)

        ds_full = ImageListDataset(root, spec.samples, transform=None)

        n = len(ds_full)
        if n == 0:
            raise RuntimeError(f"[data_factory] No images for spec cid={getattr(spec,'cid','?')} at root={root}")

        # 分层切分（按目录名→label）
        label_map = _build_label_map_from_samples(root, spec.samples)
        all_labels = []
        for rp in spec.samples:
            p = os.path.join(root, rp)
            class_name = os.path.basename(os.path.dirname(p))
            all_labels.append(label_map[class_name])

        tr_idx, val_idx = _stratified_split_by_labels(
            all_labels,
            val_ratio=val_ratio,
            seed=2025 + int(getattr(spec, "cid", 0))
        )
        te_idx = val_idx  

        tr_set, val_set, te_set = Subset(ds_full, tr_idx), Subset(ds_full, val_idx), Subset(ds_full, te_idx)

        def collate_fn(batch):
            imgs, class_names = zip(*batch)
            enc = tok_or_proc(images=list(imgs), return_tensors="pt")
            labels = torch.tensor([label_map[c] for c in class_names], dtype=torch.long)
            enc["labels"] = labels
            return enc

        return (
            DataLoader(tr_set, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_fn),
            DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn),
            DataLoader(te_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn),
        )

    elif dataset_name == "domainnet_hf":
        root = cfg["dataset"]["root"]
        _ensure_dir_exists(root)

        ds_full = DomainNetHFDataset(root, spec.samples)

        n = len(ds_full)
        if n == 0:
            raise RuntimeError(f"[data_factory] No images for spec cid={getattr(spec,'cid','?')} at root={root}")

        all_labels = _load_labels_for_domainnet_hf(root, spec.samples)
        tr_idx, val_idx = _stratified_split_by_labels(
            all_labels,
            val_ratio=val_ratio,
            seed=2025 + int(getattr(spec, "cid", 0))
        )
        te_idx = val_idx

        tr_set, val_set, te_set = Subset(ds_full, tr_idx), Subset(ds_full, val_idx), Subset(ds_full, te_idx)

        def collate_fn(batch):
            # batch: List[(PIL.Image, int_label)]
            imgs, labels = zip(*batch)
            enc = tok_or_proc(images=list(imgs), return_tensors="pt")
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
            return enc

        # 先保稳：单进程读取 parquet
        return (
            DataLoader(
                tr_set, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=False, persistent_workers=False,
                collate_fn=collate_fn
            ),
            DataLoader(
                val_set, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False, persistent_workers=False,
                collate_fn=collate_fn
            ),
            DataLoader(
                te_set, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False, persistent_workers=False,
                collate_fn=collate_fn
            ),
        )


    elif dataset_name == "amazon_md":
        amazon_cfg = cfg["dataset"]["amazon"]
        text_field = amazon_cfg.get("text_field", "text")
        label_field = amazon_cfg.get("label_field", "label")

        ds_full = AmazonTagDataset(spec.samples, text_field=text_field, label_field=label_field)

        n = len(ds_full)
        if n == 0:
            raise RuntimeError(f"[data_factory] No text rows for spec cid={getattr(spec,'cid','?')}")


        all_labels = []
        for tag in spec.samples:
            path, idx_str = tag.split("#", 1)

            pass


        n_val = max(1, n // 5)
        idx_all = list(range(n))
        rng = random.Random(2025 + int(getattr(spec, "cid", 0)))
        rng.shuffle(idx_all)
        val_idx = sorted(idx_all[:n_val])
        tr_idx = sorted(idx_all[n_val:])
        te_idx = val_idx

        tr_set, val_set, te_set = Subset(ds_full, tr_idx), Subset(ds_full, val_idx), Subset(ds_full, te_idx)

        def collate_fn(batch):
            texts, labels = zip(*batch)
            enc = tok_or_proc(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=_pick_int(cfg.get("data", {}), "max_length", 256)
            )
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
            return enc

        return (
            DataLoader(tr_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True, collate_fn=collate_fn),
            DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn),
            DataLoader(te_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn),
        )

    elif dataset_name == "pkl_partition":
        root = cfg["dataset"]["root"]
        cid = int(getattr(spec, "cid", 0))
        batch_size = int(train_cfg.get("batch_size", 32))
        val_ratio = float(data_cfg.get("val_ratio", 0.2))

        tr_set = _load_client_pkl(root, cid, "train")
        va_set = _load_client_pkl(root, cid, "val")
        te_set = _load_client_pkl(root, cid, "test")

        if tr_set is None:
            raise RuntimeError(f"[data_factory] train pkl not found for client{cid} under {root}")

        if va_set is None or te_set is None:

            labels = []
            for _, y in tr_set:
                if torch.is_tensor(y):
                    labels.append(int(y.item()))
                else:
                    labels.append(int(y))
            tr_idx, val_idx = _stratified_split_by_labels(
                labels,
                val_ratio=val_ratio,
                seed=2025 + cid
            )
            full_indices = list(range(len(tr_set)))
            te_idx = val_idx 

            tr_set = Subset(tr_set, tr_idx)
            va_set = Subset(tr_set.dataset, val_idx) if va_set is None else va_set
            te_set = Subset(tr_set.dataset, te_idx) if te_set is None else te_set

 
        def collate_fn(batch):
            xs, ys = zip(*batch)

            enc = tok_or_proc(images=list(xs), return_tensors="pt")
            labels = []
            for y in ys:
                if torch.is_tensor(y):
                    labels.append(int(y.item()))
                else:
                    labels.append(int(y))
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
            return enc

        return (
            DataLoader(tr_set, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True, collate_fn=collate_fn),
            DataLoader(va_set, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True, collate_fn=collate_fn),
            DataLoader(te_set, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True, collate_fn=collate_fn),
        )
    else:
        raise ValueError(f"[data_factory] Unsupported dataset: {dataset_name}")
