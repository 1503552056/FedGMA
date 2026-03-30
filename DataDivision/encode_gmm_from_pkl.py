

import os
import sys
import glob
import argparse
from collections import Counter
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def pil_or_nd_to_tensor(x) -> torch.Tensor:
    """PIL / numpy / tensor -> float32 Tensor[C,H,W] in [0,1] for TF.to_tensor()."""
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
    """Wrap an existing dataset to override/ensure a specific transform on images."""
    def __init__(self, base: Dataset, transform: Optional[T.Compose] = None):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

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
    body = nn.Sequential(*list(m.children())[:-1])  # [N, 512, 1, 1]
    encoder = nn.Sequential(body, nn.Flatten())     # [N, 512]
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def extract_features(
    ds: Dataset,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = encoder(x)  # [B, D]
        feats.append(z.cpu().numpy())
        labels.append(y.numpy())
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)
    y = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64)
    return X, y


def _extract_covariances(gmm: GaussianMixture) -> np.ndarray:
    ct = gmm.covariance_type
    if ct in ("full", "diag"):
        return gmm.covariances_.copy()
    elif ct == "tied":
        C = gmm.covariances_
        G = gmm.weights_.shape[0]
        return np.stack([C.copy() for _ in range(G)], axis=0)  # (G, D, D)
    elif ct == "spherical":
        G = gmm.weights_.shape[0]
        D = gmm.means_.shape[1]
        sig2 = gmm.covariances_.reshape(G, 1, 1)
        return np.eye(D)[None, ...] * sig2
    else:
        raise ValueError(f"Unsupported covariance_type: {ct}")


def fit_gmm_per_class(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 3,
    covariance_type: str = "diag",
    reg_covar: float = 1e-5,
    max_iter: int = 200,
    tol: float = 1e-3,
    random_state: int = 42
) -> Dict[int, Dict[str, Any]]:
    """Fit GMM per class, skip classes with <= 1 sample."""
    models: Dict[int, Dict[str, Any]] = {}
    classes = np.unique(y)

    for cls in classes:
        Xk = X[y == cls]
        n_k, D = Xk.shape

        if n_k <= 1:
            print(f"[SKIP] label {cls}: only {n_k} sample(s), skip GMM fitting")
            continue

        n_comp = min(n_components, n_k)
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            init_params="kmeans",
        )
        gmm.fit(Xk)

        models[int(cls)] = {
            "weights": gmm.weights_.copy(),
            "means": gmm.means_.copy(),
            "covariances": _extract_covariances(gmm),
            "covariance_type": covariance_type,
            "n_samples": int(n_k),
        }
        print(f"[OK] label {cls}: n={n_k}, G={gmm.weights_.shape[0]}, D={D}, cov='{covariance_type}'")

    return models



def maybe_pca(X: np.ndarray, pca_dim: int, whiten: bool, seed: int) -> Tuple[np.ndarray, Optional[PCA]]:
    if pca_dim is None or pca_dim <= 0 or pca_dim >= X.shape[1]:
        return X, None
    pca = PCA(n_components=pca_dim, whiten=whiten, random_state=seed)
    Xp = pca.fit_transform(X)
    print(f"[PCA] {X.shape[1]} → {Xp.shape[1]} (whiten={whiten})")
    return Xp, pca


def save_gmm_npz(output_npz: str,
                 label_models: Dict[int, Dict[str, Any]],
                 pca: Optional[PCA],
                 meta: Dict[str, Any]):
    out = {}
    for lbl, m in label_models.items():
        out[f"{lbl}_weights"] = m["weights"]
        out[f"{lbl}_means"] = m["means"]
        out[f"{lbl}_covariances"] = m["covariances"]
        out[f"{lbl}_n"] = np.array([m["n_samples"]], dtype=np.int64)
        out[f"{lbl}_covariance_type"] = np.array([m["covariance_type"]], dtype=object)
    out["labels_sorted"] = np.array(sorted(label_models.keys()), dtype=np.int64)
    for k, v in meta.items():
        out[f"meta_{k}"] = np.array([v], dtype=object)
    if pca is not None:
        out["pca_components_"] = pca.components_
        out["pca_mean_"] = pca.mean_
        out["pca_whiten"] = np.array([pca.whiten], dtype=object)
        if hasattr(pca, "explained_variance_"):
            out["pca_explained_variance_"] = pca.explained_variance_
        if hasattr(pca, "explained_variance_ratio_"):
            out["pca_explained_variance_ratio_"] = pca.explained_variance_ratio_
    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
    np.savez(output_npz, **out)
    print(f"[SAVE] {output_npz}")


def quick_preview_labels(ds: Dataset, k: int = 1000):
    labels = []
    K = min(k, len(ds))
    for i in range(K):
        _, yi = ds[i]
        yi = int(yi.item()) if torch.is_tensor(yi) else int(yi)
        labels.append(yi)
    cnt = Counter(labels)
    print(f"[INFO] preview labels (<= {K}): {dict(sorted(cnt.items()))}")


# ------------------------------- Batch Main -------------------------------

def process_one_pkl(pkl_path: str,
                    out_dir: str,
                    encoder: nn.Module,
                    device: torch.device,
                    resize: int,
                    batch_size: int,
                    num_workers: int,
                    gmm_components: int,
                    covariance_type: str,
                    reg_covar: float,
                    max_iter: int,
                    tol: float,
                    seed: int,
                    pca_dim: int,
                    pca_whiten: bool,
                    clear_base_transform: bool,
                    overwrite: bool) -> None:
    try:
        print(f"\n[LOAD] {pkl_path}")
        base_ds = torch.load(pkl_path, weights_only=False)

        if clear_base_transform and hasattr(base_ds, "transform"):
            base_ds.transform = None

        tfm = imagenet_eval_transform(resize_to=resize)
        ds = TransformedDataset(base=base_ds, transform=tfm)

        quick_preview_labels(base_ds, k=500)


        X, y = extract_features(
            ds, encoder, device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda")
        )
        print(f"[FEATS] X={X.shape}, y={y.shape}, labels={sorted(np.unique(y).tolist())}")

        # PCA
        Xp, pca = maybe_pca(X, pca_dim=pca_dim, whiten=pca_whiten, seed=seed)

        # GMM
        models = fit_gmm_per_class(
            Xp, y,
            n_components=gmm_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            max_iter=max_iter,
            tol=tol,
            random_state=seed
        )


        base = os.path.splitext(os.path.basename(pkl_path))[0] + ".npz"
        out_path = os.path.join(out_dir, base)
        if os.path.exists(out_path) and not overwrite:
            print(f"[SKIP] exists: {out_path}")
            return

        meta = dict(
            src_pkl=pkl_path,
            encoder="resnet18_imagenet",
            resize=resize,
            pca_dim=(Xp.shape[1] if pca is not None else 0),
            covariance_type=covariance_type,
            gmm_components=gmm_components,
            reg_covar=reg_covar,
            max_iter=max_iter,
            tol=tol,
            seed=seed
        )
        save_gmm_npz(out_path, models, pca, meta)

    except Exception as e:
        print(f"[ERROR] {pkl_path}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Batch encode *.pkl into per-class GMM npz files.")
    ap.add_argument("--pkl_dir", type=str, required=True, help="Folder containing *.pkl")
    ap.add_argument("--out_dir", type=str, required=True, help="Folder to save *.npz")
    ap.add_argument("--pattern", type=str, default="*.pkl", help="Glob pattern for pkls")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)

    # GMM / PCA
    ap.add_argument("--gmm_components", type=int, default=3)
    ap.add_argument("--covariance_type", type=str, default="diag", choices=["diag", "full", "tied", "spherical"])
    ap.add_argument("--reg_covar", type=float, default=1e-5)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pca_dim", type=int, default=0)
    ap.add_argument("--pca_whiten", action="store_true")

    # Practical toggles
    ap.add_argument("--clear_base_transform", action="store_true",
                    help="If set, clear base_ds.transform to avoid double-normalization.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing npz")
    ap.add_argument("--project_root", type=str, default="",
                    help="If provided, will be inserted into sys.path for torch.load() class import (e.g., dataset_deal).")
    args = ap.parse_args()


    if args.project_root:
        proj_root = os.path.abspath(args.project_root)
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)
        try:
            import dataset_deal.basic_dataset  
        except Exception as e:
            print(f"[WARN] cannot import dataset_deal.basic_dataset from {proj_root}: {e}")


    device = torch.device(args.device)
    print(f"[INFO] device={device}")
    encoder = build_encoder_resnet18(device)

    files = sorted(glob.glob(os.path.join(args.pkl_dir, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No pkl found in {args.pkl_dir} (pattern={args.pattern})")

    print(f"[INFO] found {len(files)} pkls:")
    for i, f in enumerate(files):
        print(f"  [{i}] {os.path.basename(f)}")


    for pkl_path in files:
        process_one_pkl(
            pkl_path=pkl_path,
            out_dir=args.out_dir,
            encoder=encoder,
            device=device,
            resize=args.resize,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gmm_components=args.gmm_components,
            covariance_type=args.covariance_type,
            reg_covar=args.reg_covar,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
            pca_dim=args.pca_dim,
            pca_whiten=args.pca_whiten,
            clear_base_transform=args.clear_base_transform,
            overwrite=args.overwrite
        )


if __name__ == "__main__":
    main()
