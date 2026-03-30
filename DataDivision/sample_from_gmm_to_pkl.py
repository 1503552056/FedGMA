
import os
import math
import argparse
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as TF
import torchvision.models as models

# -------------------- Encoder / transforms --------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_encoder_resnet18(device: torch.device) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    body = nn.Sequential(*list(m.children())[:-1])  # [N,512,1,1]
    enc = nn.Sequential(body, nn.Flatten())         # [N,512]
    enc.eval().to(device)
    for p in enc.parameters(): p.requires_grad = False
    return enc

def inv_preprocess(img_01: torch.Tensor, resize_to: int = 224) -> torch.Tensor:
    """x in [0,1] -> normalize to ImageNet stats."""
    x = TF.resize(img_01, [resize_to, resize_to], antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1,3,1,1)
    return (x - mean) / std

# -------------------- GMM I/O & PCA inverse --------------------

def load_gmm_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data["labels_sorted"].astype(int).tolist()
    per_label = {}
    label_counts = {}
    for lbl in labels:
        per_label[lbl] = dict(
            weights=data[f"{lbl}_weights"],
            means=data[f"{lbl}_means"],
            covs=data[f"{lbl}_covariances"],
            cov_type=str(data[f"{lbl}_covariance_type"][0]),
        )
        label_counts[lbl] = int(data[f"{lbl}_n"][0])
    pca = None
    if "pca_components_" in data.files and "pca_mean_" in data.files:
        comps = data["pca_components_"]
        mean  = data["pca_mean_"]
        whit  = bool(data["pca_whiten"][0]) if "pca_whiten" in data.files else False
        var   = data.get("pca_explained_variance_", None)
        pca = dict(components=comps, mean=mean, whiten=whit, explained_variance=var)
    return dict(labels=labels, per_label=per_label, label_counts=label_counts, pca=pca)

def inverse_pca_means_covs(means, covs, cov_type, pca):
    comps = pca["components"]; mean0 = pca["mean"]
    whiten = bool(pca.get("whiten", False))
    var = pca.get("explained_variance", None)
    M, D = comps.shape
    if whiten:
        if var is None: raise ValueError("PCA whiten=True but missing explained_variance_.")
        s = np.sqrt(var)
        A = (comps.T * s).T
    else:
        A = comps
    means_x = means @ A + mean0
    # cov to full in PCA space
    if cov_type == "diag":
        covs_z = np.array([np.diag(c) for c in covs], dtype=float)
    elif cov_type in ("full",):
        covs_z = covs.astype(float)
    else:
        if covs.ndim == 3 and covs.shape[1] == M:
            covs_z = covs.astype(float)
        else:
            raise ValueError(f"Unsupported cov_type: {cov_type}")
    AT = A.T
    covs_x = np.empty((covs_z.shape[0], D, D), dtype=float)
    for g in range(covs_z.shape[0]):
        covs_x[g] = AT @ covs_z[g] @ A
    return means_x, covs_x

def gmm_to_encoder_space(model):
    labels = model["labels"]; per = model["per_label"]; pca = model["pca"]
    out = {}
    for lbl in labels:
        w, m, C, ct = per[lbl]["weights"], per[lbl]["means"], per[lbl]["covs"], per[lbl]["cov_type"]
        if pca is not None:
            m_x, C_x = inverse_pca_means_covs(m, C, ct, pca)
        else:
            if ct == "diag":
                C_x = np.array([np.diag(ci) for ci in C], dtype=float); m_x = m.astype(float)
            elif ct in ("full",):
                C_x = C.astype(float); m_x = m.astype(float)
            else:
                if C.ndim == 3 and C.shape[1] == m.shape[1]:
                    C_x = C.astype(float); m_x = m.astype(float)
                else:
                    raise ValueError(f"Unsupported cov_type: {ct}")
        out[lbl] = dict(weights=w.astype(float), means=m_x, covs=C_x)
    return labels, out

# -------------------- Sampling targets in feature space --------------------

def sample_from_gaussian(mean: np.ndarray, cov: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    D = mean.shape[0]
    cov_reg = cov + np.eye(D) * 1e-9
    return rng.multivariate_normal(mean, cov_reg).astype(np.float32)

def plan_counts_by_ratio(label_counts: Dict[int, int], sample_ratio: float) -> Dict[int, int]:
    total = sum(label_counts.values())
    target_total = max(1, int(round(total * sample_ratio)))
    ratios = {l: c / total for l, c in label_counts.items()}
    counts = {l: int(round(r * target_total)) for l, r in ratios.items()}
    # 修正四舍五入误差
    diff = target_total - sum(counts.values())
    if diff != 0:
        # 按最大余数或顺序调整
        keys = sorted(counts.keys())
        for k in (keys if diff > 0 else reversed(keys)):
            counts[k] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            if diff == 0: break
    return counts

def build_targets(per_label_enc: Dict[int, Dict[str, np.ndarray]],
                  mode: str,
                  per_class: Optional[int],
                  label_counts: Dict[int, int],
                  sample_ratio: Optional[float],
                  total_samples: Optional[int],
                  seed: int) -> List[Tuple[int, np.ndarray]]:
    """
    返回 (label, z_target) 列表（encoder 特征空间的目标向量）。
    生成策略：
      - mode='means': 每个类取所有分量均值
      - mode='sample': 采样数量由 per_class 或 sample_ratio/total_samples 决定
    """
    rng = np.random.RandomState(seed)
    targets: List[Tuple[int, np.ndarray]] = []

    if mode == "means":
        for lbl, g in per_label_enc.items():
            for gi in range(len(g["weights"])):
                targets.append((lbl, g["means"][gi].astype(np.float32)))
        return targets

    # mode == 'sample'
    # 决定每类采样数
    if per_class is not None and per_class > 0:
        per_label_num = {l: per_class for l in per_label_enc.keys()}
    elif total_samples is not None and total_samples > 0:
        total = int(total_samples)
        total_ratio = sum(label_counts.values())
        ratios = {l: label_counts[l] / total_ratio for l in per_label_enc.keys()}
        per_label_num = {l: int(round(ratios[l] * total)) for l in per_label_enc.keys()}
        # 纠正误差
        diff = total - sum(per_label_num.values())
        for l in sorted(per_label_num.keys()):
            if diff == 0: break
            per_label_num[l] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
    elif sample_ratio is not None and sample_ratio > 0:
        per_label_num = plan_counts_by_ratio(label_counts, sample_ratio)
    else:
        raise ValueError("When mode=sample, you must set --per_class or --sample_ratio or --total_samples.")

    # 按各类 GMM 权重分配到各分量，再采样
    for lbl, g in per_label_enc.items():
        need = per_label_num.get(lbl, 0)
        if need <= 0: continue
        w = g["weights"]; w = w / w.sum()
        alloc = rng.multinomial(need, w)  # 每个分量采样多少
        for gi, k in enumerate(alloc):
            for _ in range(int(k)):
                z = sample_from_gaussian(g["means"][gi], g["covs"][gi], rng)
                targets.append((lbl, z))
    return targets

# -------------------- Feature inversion: optimize pixel image --------------------

def tv_loss(x):
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dh.abs().mean() + dw.abs().mean())

@torch.no_grad()
def encode_batch(x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
    return encoder(x)

def invert_one(encoder: nn.Module,
               z_target: np.ndarray,
               steps: int = 400,
               lr: float = 0.08,
               tv_weight: float = 1e-3,
               l2_weight: float = 1e-4,
               init: str = "noise",
               device: str = "cuda",
               out_size: int = 224,
               log_every: int = 100) -> torch.Tensor:
    """
    返回 [3,H,W] in [0,1]
    """
    device = torch.device(device)
    encoder.eval()
    if init == "zeros":
        img = torch.zeros(1,3,out_size,out_size, device=device, requires_grad=True)
    else:
        img = torch.rand(1,3,out_size,out_size, device=device, requires_grad=True)

    z_t = torch.tensor(z_target, device=device, dtype=torch.float32).view(1,-1)
    opt = torch.optim.Adam([img], lr=lr)

    for t in range(steps):
        opt.zero_grad()
        x_norm = inv_preprocess(img.clamp(0,1), resize_to=out_size)
        z = encoder(x_norm)
        loss_feat = F.mse_loss(z, z_t)
        loss = loss_feat + tv_weight * tv_loss(img) + l2_weight * ((img - 0.5)**2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            img.clamp_(0,1)
        if (t+1) % log_every == 0:
            print(f"  step {t+1}/{steps} | feat={loss_feat.item():.4e} total={loss.item():.4e}")
    return img.detach().squeeze(0).cpu()

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Sample images from GMM and save as training-ready PKL.")
    ap.add_argument("--gmm", type=str, required=True, help="Path to GMM npz")
    ap.add_argument("--out_pkl", type=str, required=True, help="Output PKL path")
    # 采样策略
    ap.add_argument("--mode", type=str, default="sample", choices=["sample","means"],
                    help="'sample' to draw from GMM; 'means' to use each component mean once")
    ap.add_argument("--per_class", type=int, default=None, help="If set, sample this many per class (mode=sample)")
    ap.add_argument("--sample_ratio", type=float, default=None, help="Total = ratio * original_count (mode=sample)")
    ap.add_argument("--total_samples", type=int, default=None, help="Total fixed number to sample (mode=sample)")
    # 反演参数
    ap.add_argument("--resize", type=int, default=224, help="Output image size (HxW)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--tv", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--init", type=str, default="noise", choices=["noise","zeros"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pkl)), exist_ok=True)

    device = torch.device(args.device)
    encoder = build_encoder_resnet18(device)

    # 1) 读 GMM 并转到 encoder 空间
    gmm = load_gmm_npz(args.gmm)
    labels, per_label_enc = gmm_to_encoder_space(gmm)
    label_counts = gmm["label_counts"]

    # 2) 组装特征目标
    targets = build_targets(per_label_enc,
                            mode=args.mode,
                            per_class=args.per_class,
                            label_counts=label_counts,
                            sample_ratio=args.sample_ratio,
                            total_samples=args.total_samples,
                            seed=123)
    print(f"[INFO] Targets: {len(targets)}  (labels covered: {sorted(set(l for l,_ in targets))})")

    # 3) 逐个特征做像素反演
    imgs: List[torch.Tensor] = []
    labs: List[int] = []
    for i, (lbl, zt) in enumerate(targets):
        print(f"\n[INV] {i+1}/{len(targets)} label={lbl}")
        img = invert_one(encoder, zt,
                         steps=args.steps, lr=args.lr,
                         tv_weight=args.tv, l2_weight=args.l2,
                         init=args.init, device=device.type,
                         out_size=args.resize, log_every=max(50, args.steps//10))
        imgs.append(img)   # [3,H,W] in [0,1]
        labs.append(int(lbl))

    if len(imgs) == 0:
        raise RuntimeError("No images generated. Check your sampling parameters.")

    X = torch.stack(imgs, dim=0)  # [N,3,H,W]
    Y = torch.tensor(labs, dtype=torch.long)

    # 4) 保存为可训练 PKL
    out_obj = {"x": X, "y": Y}
    torch.save(out_obj, args.out_pkl)
    print(f"\n[SAVED] {args.out_pkl}")
    print(f"  x: {tuple(X.shape)}  in [0,1]")
    print(f"  y: {tuple(Y.shape)}  classes: {sorted(set(labs))}")

if __name__ == "__main__":
    main()
