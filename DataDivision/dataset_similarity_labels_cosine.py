
import os
import sys
import glob
import csv
import argparse
from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def pil_or_nd_to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    try:
        return TF.to_tensor(x).float()
    except Exception:
        arr = np.asarray(x)
        if arr.ndim == 3:
            return torch.from_numpy(arr).permute(2, 0, 1).float()
        return torch.from_numpy(arr).unsqueeze(0).float()

class TransformedDataset(Dataset):
    def __init__(self, base: Dataset, transform: Optional[T.Compose] = None):
        self.base = base
        self.transform = transform

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if not isinstance(img, torch.Tensor):
            img = pil_or_nd_to_tensor(img)
        if self.transform is not None:
            img = self.transform(img)
        label = int(label.item()) if torch.is_tensor(label) else int(label)
        return img, label

def imagenet_eval_transform(resize_to: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((resize_to, resize_to), antialias=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def build_encoder_resnet18(device: torch.device) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    body = nn.Sequential(*list(m.children())[:-1]) 
    enc = nn.Sequential(body, nn.Flatten())        
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad = False
    return enc



def histogram_labels(ds: Dataset, K: Optional[int]=None) -> np.ndarray:
    """
    Count labels of a dataset. If K is None, infer as max(label)+1 seen in ds.
    Return a length-K histogram (float).
    """
    labels = []
    for i in range(len(ds)):
        _, y = ds[i]
        y = int(y.item()) if torch.is_tensor(y) else int(y)
        labels.append(y)
    if K is None:
        K = (max(labels) + 1) if labels else 0
    hist = np.zeros(K, dtype=float)
    for y in labels:
        if 0 <= y < K:
            hist[y] += 1.0
    return hist

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float=1e-12) -> float:
    """
    Jensen-Shannon Divergence using natural log (range: [0, ln2])
    """
    p = p.astype(float); q = q.astype(float)
    p_sum = p.sum(); q_sum = q.sum()
    if p_sum <= 0 and q_sum <= 0:
        return 0.0
    if p_sum <= 0:
 
        qn = q / (q_sum + eps)
        m = 0.5 * qn
        kl_q_m = np.sum(qn * (np.log(qn + eps) - np.log(m + eps)))
        return 0.5 * kl_q_m  
    if q_sum <= 0:
        pn = p / (p_sum + eps)
        m = 0.5 * pn
        kl_p_m = np.sum(pn * (np.log(pn + eps) - np.log(m + eps)))
        return 0.5 * kl_p_m
    pn = p / (p_sum + eps)
    qn = q / (q_sum + eps)
    m = 0.5 * (pn + qn)
    kl_p_m = np.sum(pn * (np.log(pn + eps) - np.log(m + eps)))
    kl_q_m = np.sum(qn * (np.log(qn + eps) - np.log(m + eps)))
    return 0.5 * (kl_p_m + kl_q_m)

def jsd_similarity(p: np.ndarray, q: np.ndarray, eps: float=1e-12) -> float:
    """
    Map JSD to [0,1]: S = 1 - JSD / ln(2).
    """
    jsd = js_divergence(p, q, eps=eps)
    return float(max(0.0, 1.0 - jsd / np.log(2.0)))




@torch.no_grad()
def dataset_feature_centroid(
    ds: Dataset,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int=256,
    num_workers: int=2,
    pin_memory: bool=True
) -> np.ndarray:
    """
    Compute a dataset-level mean feature (normalize per-sample, average, then normalize again).
    Return a 1D numpy vector (unit L2).
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    sum_vec = None
    n = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z = encoder(x)                   
        z = torch.nn.functional.normalize(z, dim=1)  
        if sum_vec is None:
            sum_vec = z.sum(dim=0)
        else:
            sum_vec += z.sum(dim=0)
        n += z.shape[0]
    if n == 0:
        return np.zeros((encoder[-1].out_features if hasattr(encoder[-1], "out_features") else 512,), dtype=np.float32)
    mean_vec = (sum_vec / max(1, n)).cpu().numpy()
    norm = np.linalg.norm(mean_vec) + 1e-12
    return (mean_vec / norm).astype(np.float32)

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u) + 1e-12
    nv = np.linalg.norm(v) + 1e-12
    return float(np.dot(u, v) / (nu * nv))




def save_matrix_csv(path: str, M: np.ndarray, names: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file"] + names)
        for name, row in zip(names, M):
            w.writerow([name] + [f"{v:.8f}" for v in row])

def row_l1_normalize(S: np.ndarray, exclude_self: bool=False) -> np.ndarray:
    print(exclude_self)
    S = S.copy().astype(float)
    if exclude_self:
        np.fill_diagonal(S, 0.0)
    row_sum = np.sum(np.abs(S), axis=1, keepdims=True) + 1e-12
    return S / row_sum




def main():
    ap = argparse.ArgumentParser(description="Pairwise dataset similarities (labels JSD & feature cosine) over *.pkl in a folder.")
    ap.add_argument("--pkl_dir", type=str, required=True, help="Folder containing *.pkl datasets")
    ap.add_argument("--pattern", type=str, default="*.pkl", help="Glob pattern for pkl files")
    ap.add_argument("--out_prefix", type=str, default="./out/ds", help="Output prefix (CSV/NPY will be created)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--clear_base_transform", action="store_true", help="Clear base_ds.transform to avoid double-normalization")
    ap.add_argument("--project_root", type=str, default="", help="Insert into sys.path for torch.load custom classes (e.g., dataset_deal)")
    ap.add_argument("--exclude_self", action="store_true", help="Set diagonal to 0 before row-normalization (optional)")
    ap.add_argument("--num_classes", type=int, default=-1, help="If >0, force label histogram length; else inferred from data")
    args = ap.parse_args()

    if args.project_root:
        pr = os.path.abspath(args.project_root)
        if pr not in sys.path:
            sys.path.insert(0, pr)
        try:
            import dataset_deal.basic_dataset  
        except Exception as e:
            print(f"[WARN] cannot import dataset_deal.basic_dataset from {pr}: {e}")

    device = torch.device(args.device)
    print(f"[INFO] device={device}")

    files = sorted(glob.glob(os.path.join(args.pkl_dir, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No pkl found in {args.pkl_dir} (pattern={args.pattern})")
    names = [os.path.basename(p) for p in files]
    print(f"[INFO] found {len(files)} pkls:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")


    encoder = build_encoder_resnet18(device)
    tfm = imagenet_eval_transform(resize_to=args.resize)


    label_hists: List[np.ndarray] = []
    centroids: List[np.ndarray] = []
    max_label_seen = -1

    for p in files:
        print(f"\n[LOAD] {p}")
        base_ds = torch.load(p, weights_only=False)
        if args.clear_base_transform and hasattr(base_ds, "transform"):
            base_ds.transform = None
        ds = TransformedDataset(base_ds, transform=tfm)


        labels = []
        for i in range(len(base_ds)):
            _, yi = base_ds[i]
            yi = int(yi.item()) if torch.is_tensor(yi) else int(yi)
            labels.append(yi)
        if labels:
            max_label_seen = max(max_label_seen, max(labels))
        label_hists.append(np.array(labels, dtype=int))  
        c = dataset_feature_centroid(ds, encoder, device, args.batch_size, args.num_workers, pin_memory=(device.type=="cuda"))
        centroids.append(c)


    if args.num_classes and args.num_classes > 0:
        K = int(args.num_classes)
    else:
        K = int(max_label_seen + 1) if max_label_seen >= 0 else 0
    print(f"\n[INFO] Using num_classes = {K} for label histograms")

    hist_arrays: List[np.ndarray] = []
    for labels in label_hists:
        hist = np.zeros((K,), dtype=float)
        if labels.size > 0:
            for y in labels:
                if 0 <= y < K:
                    hist[y] += 1.0
        hist_arrays.append(hist)


    n = len(files)
    S_label = np.zeros((n, n), dtype=float)
    S_cos   = np.zeros((n, n), dtype=float)

    for i in range(n):
        S_label[i, i] = 1.0
        S_cos[i, i]   = 1.0
        for j in range(i+1, n):
            sij = jsd_similarity(hist_arrays[i], hist_arrays[j])
            S_label[i, j] = S_label[j, i] = sij

            cij = cosine_similarity(centroids[i], centroids[j])
            cij = max(-1.0, min(1.0, cij))
            cij01 = 0.5 * (cij + 1.0)
            S_cos[i, j] = S_cos[j, i] = cij01

    print("exclude_self:", args.exclude_self)
    S_label_row = row_l1_normalize(S_label, exclude_self=args.exclude_self)
    S_cos_row   = row_l1_normalize(S_cos,   exclude_self=args.exclude_self)

    show = min(5, n)
    print("\n[Label JSD similarity] top-left (raw):")
    print(np.round(S_label[:show, :show], 4))
    print("\n[Label JSD similarity] top-left (row L1):")
    print(np.round(S_label_row[:show, :show], 4))

    print("\n[Feature cosine similarity] top-left (raw):")
    print(np.round(S_cos[:show, :show], 4))
    print("\n[Feature cosine similarity] top-left (row L1):")
    print(np.round(S_cos_row[:show, :show], 4))

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    np.save(f"{args.out_prefix}_label_similarity.npy", S_label)
    np.save(f"{args.out_prefix}_label_similarity_rownorm.npy", S_label_row)
    save_matrix_csv(f"{args.out_prefix}_label_similarity.csv", S_label, names)
    save_matrix_csv(f"{args.out_prefix}_label_similarity_rownorm.csv", S_label_row, names)

    np.save(f"{args.out_prefix}_cosine_similarity.npy", S_cos)
    np.save(f"{args.out_prefix}_cosine_similarity_rownorm.npy", S_cos_row)
    save_matrix_csv(f"{args.out_prefix}_cosine_similarity.csv", S_cos, names)
    save_matrix_csv(f"{args.out_prefix}_cosine_similarity_rownorm.csv", S_cos_row, names)

    print(f"\n[SAVED]")
    print(f"  {args.out_prefix}_label_similarity(.npy/.csv)")
    print(f"  {args.out_prefix}_label_similarity_rownorm(.npy/.csv)")
    print(f"  {args.out_prefix}_cosine_similarity(.npy/.csv)")
    print(f"  {args.out_prefix}_cosine_similarity_rownorm(.npy/.csv)")


if __name__ == "__main__":
    main()
