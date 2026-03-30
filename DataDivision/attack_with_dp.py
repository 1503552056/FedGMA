

import os
import sys
import csv
import math
import argparse
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.utils import save_image


try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


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
    def __init__(self, base: Dataset, transform: Optional[T.Compose]):
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

class RawDataset(Dataset):
    def __init__(self, base: Dataset, to_size: Optional[int] = None):
        self.base = base
        self.to_size = to_size
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        if not isinstance(img, torch.Tensor):
            img = pil_or_nd_to_tensor(img)
        if self.to_size is not None:
            img = TF.resize(img, [self.to_size, self.to_size], antialias=True)
        return img.clamp(0,1), int(label.item()) if torch.is_tensor(label) else int(label)

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
    for p in enc.parameters(): p.requires_grad = False
    return enc


def load_gmm_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data["labels_sorted"].astype(int).tolist()
    per_label = {}
    for lbl in labels:
        per_label[lbl] = dict(
            weights=data[f"{lbl}_weights"],
            means=data[f"{lbl}_means"],
            covs=data[f"{lbl}_covariances"],
            cov_type=str(data[f"{lbl}_covariance_type"][0]),
            n=int(data[f"{lbl}_n"][0]) if f"{lbl}_n" in data.files else int(data.get("n", [0])[0]) if "n" in data.files else 0,
        )
    pca = None
    if "pca_components_" in data.files and "pca_mean_" in data.files:
        comps = data["pca_components_"]
        mean  = data["pca_mean_"]
        whit  = bool(data["pca_whiten"][0]) if "pca_whiten" in data.files else False
        var   = data.get("pca_explained_variance_", None)
        pca = dict(components=comps, mean=mean, whiten=whit, explained_variance=var)
    return dict(labels=labels, per_label=per_label, pca=pca)

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
        n_val = per[lbl].get("n", 0)
        out[lbl] = dict(weights=w.astype(float), means=m_x, covs=C_x, n=np.array([n_val], dtype=float))
    return labels, out


def apply_noise_to_gmm(prepared: Dict[int, Dict[str, np.ndarray]],
                       mu_std: float = 0.0,
                       cov_eps: float = 0.0,
                       w_std: float = 0.0,
                       rng: Optional[np.random.RandomState] = None,
                       mu_mode: str = "per-dim",   
                       mu_coeff: float = 1.0,
                       inflate_cov_s: float = 1.0
                      ) -> Dict[int, Dict[str, np.ndarray]]:
    if rng is None:
        rng = np.random.RandomState()
    out = {}
    for lbl, g in prepared.items():
        w = g["weights"].astype(float).copy()
        m = g["means"].astype(float).copy()
        C = g["covs"].astype(float).copy()
        G, D = m.shape
        if inflate_cov_s and inflate_cov_s > 1.0:
            C *= float(inflate_cov_s)**2
        if mu_std > 0 or (mu_mode == "mahal" and mu_coeff > 0):
            for gi in range(G):
                diag_var = np.clip(np.diag(C[gi]), 1e-12, None)
                std_vec = np.sqrt(diag_var)
                if mu_mode == "isotropic":
                    m[gi] += rng.normal(0.0, mu_std, size=D)
                elif mu_mode == "per-dim":
                    m[gi] += rng.normal(0.0, mu_std, size=D) * std_vec
                elif mu_mode == "mahal":
                    m[gi] += rng.normal(0.0, mu_coeff, size=D) * std_vec
                else:
                    raise ValueError("mu_mode must be 'isotropic' | 'per-dim' | 'mahal'")
        if cov_eps > 0:
            C += np.eye(D)[None, ...] * cov_eps
        if w_std > 0:
            w = w + rng.normal(0.0, w_std, size=w.shape)
            w = np.clip(w, 1e-12, None)
        w = w / w.sum()
        out[lbl] = dict(weights=w, means=m, covs=C, n=g.get("n", np.array([0.0])))
    return out


def gaussian_noise_sigma(sens: float, eps: float, delta: float) -> float:
    return sens * math.sqrt(2.0 * math.log(1.25 / max(delta, 1e-12))) / max(eps, 1e-12)

def l2_clip_(Z: torch.Tensor, R: float) -> torch.Tensor:
    norms = Z.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    scale = (R / norms).clamp(max=1.0)
    return Z * scale

def dp_release_from_params(per_label_enc: Dict[int, Dict[str, np.ndarray]],
                           epsilon: float, delta: float, R: float,
                           spd_jitter: float = 1e-6,
                           seed: int = 0):

    if epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon, delta 必须为正数。")

    labels = list(per_label_enc.keys())
    if len(labels) == 0:
        return per_label_enc, {}


    L = len(labels)
    any_lbl = labels[0]
    G = len(per_label_enc[any_lbl]["weights"])
    M = L * G * 3

    eps_unit = epsilon / M
    delta_unit = delta / M

    sigma_N = gaussian_noise_sigma(1.0, eps_unit, delta_unit)
    sigma_S = gaussian_noise_sigma(R,   eps_unit, delta_unit)
    sigma_Q = gaussian_noise_sigma(R*R, eps_unit, delta_unit)

    rng = np.random.RandomState(seed)
    dp_out = {}

    for lbl, g in per_label_enc.items():
        w = g["weights"].astype(float)
        Mns = g["means"].astype(float)   
        Cov = g["covs"].astype(float)   

        n_total = int(g.get("n", [0])[0]) if isinstance(g.get("n", 0), (np.ndarray, list)) else int(g.get("n", 0))
        if n_total is None or n_total <= 0:
            n_total = max(int(round(w.sum())), 1)
        Nk = w * (n_total / max(w.sum(), 1e-12))  

        Gc, D = Mns.shape
        new_w, new_mu, new_cov = [], [], []

        for k in range(Gc):
            mu = Mns[k]             
            Sigma = Cov[k]          
            Nk0 = float(Nk[k])

            Sk = Nk0 * mu                          
            Qk = Nk0 * (Sigma + np.outer(mu, mu))   

            Nk_tilde = Nk0 + rng.normal(0.0, sigma_N)
            Sk_tilde = Sk  + rng.normal(0.0, sigma_S, size=D)
            Qk_tilde = Qk  + rng.normal(0.0, sigma_Q, size=(D, D))

            Nk_tilde = max(Nk_tilde, 1e-6)
            mu_tilde = Sk_tilde / Nk_tilde
            Sigma_tilde = (Qk_tilde / Nk_tilde) - np.outer(mu_tilde, mu_tilde)
            Sigma_tilde = 0.5 * (Sigma_tilde + Sigma_tilde.T)
            Sigma_tilde = Sigma_tilde + np.eye(D) * spd_jitter

            new_mu.append(mu_tilde)
            new_cov.append(Sigma_tilde)
            new_w.append(Nk_tilde)

        new_w = np.clip(np.array(new_w, dtype=float), 1e-9, None)
        new_w = new_w / new_w.sum()

        dp_out[lbl] = dict(weights=new_w,
                           means=np.stack(new_mu, axis=0),
                           covs=np.stack(new_cov, axis=0),
                           n=np.array([new_w.sum()], dtype=float))

    meta = dict(
        eps=epsilon, delta=delta, clip_R=R,
        eps_unit=eps_unit, delta_unit=delta_unit,
        sigma_N=sigma_N, sigma_S=sigma_S, sigma_Q=sigma_Q,
        mechanism="Gaussian", composition="uniform-split over (N,S,Q) per component"
    )
    return dp_out, meta


def sample_from_gaussian(mean: np.ndarray, cov: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    D = mean.shape[0]
    cov_reg = cov + np.eye(D) * 1e-9
    return rng.multivariate_normal(mean, cov_reg)

def build_targets_from_gmm(per_label_enc: Dict[int, Dict[str, np.ndarray]],
                           mode: str = "means",
                           per_class: int = 1,
                           seed: int = 0) -> List[Tuple[int, np.ndarray, str]]:
    rng = np.random.RandomState(seed)
    targets = []
    for lbl, g in per_label_enc.items():
        w, M, C = g["weights"], g["means"], g["covs"]
        G = len(w)
        if mode == "means":
            for gi in range(G):
                targets.append((lbl, M[gi].copy(), f"mean_g{gi}"))
        elif mode == "sample":
            alloc = rng.multinomial(per_class, (w / w.sum()))
            for gi, k in enumerate(alloc):
                for s in range(k):
                    z = sample_from_gaussian(M[gi], C[gi], rng)
                    targets.append((lbl, z.astype(np.float32), f"s{s}_g{gi}"))
        else:
            raise ValueError("mode must be 'means' or 'sample'")
    return targets

def tv_loss(x):
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dh.abs().mean() + dw.abs().mean())

def inv_preprocess(img_01: torch.Tensor, resize_to: int = 224) -> torch.Tensor:
    x = TF.resize(img_01, [resize_to, resize_to], antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1,3,1,1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1,3,1,1)
    return (x - mean) / std

@torch.no_grad()
def encode_batch(x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
    return encoder(x)

def feature_inversion(encoder: nn.Module,
                      z_target: np.ndarray,
                      steps: int = 500,
                      lr: float = 0.1,
                      tv_weight: float = 1e-3,
                      l2_weight: float = 1e-4,
                      init: str = "noise",
                      device: str = "cuda",
                      out_size: int = 224,
                      log_every: int = 100) -> torch.Tensor:
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


@torch.no_grad()
def encode_dataset_for_eval(base_ds: Dataset,
                            encoder: nn.Module,
                            resize_to: int = 224,
                            device: str = "cuda",
                            batch_size: int = 256,
                            num_workers: int = 2):
    device = torch.device(device)
    tfm = imagenet_eval_transform(resize_to)
    ds = TransformedDataset(base_ds, tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type=="cuda"))
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = encoder(x)
        feats.append(F.normalize(z, dim=1).cpu())
        labels.append(y)
    Z = torch.cat(feats, dim=0) if feats else torch.zeros(0,512)
    Y = torch.cat(labels, dim=0) if labels else torch.zeros(0, dtype=torch.long)
    return Z, Y

def cosine_top1(z_recon: torch.Tensor, Z_db: torch.Tensor) -> Tuple[float, int]:
    zr = F.normalize(z_recon.view(1,-1), dim=1)  
    sims = torch.mv(Z_db, zr.squeeze(0))       
    val, idx = torch.max(sims, dim=0)
    return float(val.item()), int(idx.item())

def mse_psnr(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    mse = float(((x - y) ** 2).mean().item())
    if mse <= 1e-12:
        psnr = 99.0
    else:
        psnr = 10.0 * math.log10(1.0 / mse)
    return mse, psnr

def ssim_tensor(x: torch.Tensor, y: torch.Tensor) -> float:
    if not HAS_SKIMAGE:
        return float("nan")
    x_np = x.permute(1,2,0).cpu().numpy()
    y_np = y.permute(1,2,0).cpu().numpy()
    return float(ssim(x_np, y_np, data_range=1.0, channel_axis=2))


def main():
    ap = argparse.ArgumentParser(description="GMM inversion attack + evaluation (MSE/PSNR/SSIM/Cosine).")
    ap.add_argument("--gmm", type=str, required=True, help="Path to GMM npz")
    ap.add_argument("--pkl", type=str, required=True, help="Path to dataset pkl")
    ap.add_argument("--out_dir", type=str, default="./inv_out", help="Save dir")
    ap.add_argument("--mode", type=str, default="means", choices=["means","sample"],
                    help="Use GMM component means or random samples as targets")
    ap.add_argument("--per_class", type=int, default=2, help="If mode=sample, how many samples per class")
    ap.add_argument("--labels", type=str, default="", help="Comma-separated labels to attack; empty=all")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tv", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--init", type=str, default="noise", choices=["noise","zeros"])
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--project_root", type=str, default="", help="sys.path insert for torch.load custom classes")
    ap.add_argument("--dp_mode", type=str, default="none",
                    choices=["none","gaussian_mech"],
                    help="none: 关闭正式DP；gaussian_mech: 对(N,S,Q)加噪再回推")
    ap.add_argument("--epsilon", type=float, default=0.0, help="total ε for one-shot release")
    ap.add_argument("--delta", type=float, default=1e-5, help="total δ")
    ap.add_argument("--clip_R", type=float, default=10.0, help="L2 clipping radius on features/statistics")
    ap.add_argument("--spd_jitter", type=float, default=1e-6, help="Σ投影抖动以保PSD")
    ap.add_argument("--dp_mu_std", type=float, default=0.0, help="(DP-like) Gaussian std on μ in encoder space")
    ap.add_argument("--dp_cov_eps", type=float, default=0.0, help="(DP-like) add ε·I to Σ")
    ap.add_argument("--dp_w_std", type=float, default=0.0, help="(DP-like) Gaussian std on mixture weights then L1")
    ap.add_argument("--dp_mu_mode", type=str, default="per-dim", choices=["isotropic","per-dim","mahal"])
    ap.add_argument("--dp_mu_coeff", type=float, default=1.0)
    ap.add_argument("--dp_inflate_cov_s", type=float, default=1.0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.project_root:
        pr = os.path.abspath(args.project_root)
        if pr not in sys.path: sys.path.insert(0, pr)
        try:
            import dataset_deal.basic_dataset  
        except Exception as e:
            print(f"[WARN] cannot import dataset_deal from {pr}: {e}")

    device = torch.device(args.device)
    encoder = build_encoder_resnet18(device)

    print(f"[LOAD] GMM: {args.gmm}")
    model = load_gmm_npz(args.gmm)
    labels_all, per_label_enc = gmm_to_encoder_space(model)
    if args.labels:
        keep = set(int(s) for s in args.labels.split(",") if s.strip() != "")
        labels_all = [l for l in labels_all if l in keep]
        per_label_enc = {l: per_label_enc[l] for l in labels_all}

    dp_meta = None
    if args.dp_mode == "gaussian_mech":
        print(f"[DP] Central (ε,δ)-DP via Gaussian Mechanism: eps={args.epsilon}, delta={args.delta}, R={args.clip_R}")
        per_label_enc, dp_meta = dp_release_from_params(
            per_label_enc,
            epsilon=args.epsilon,
            delta=args.delta,
            R=args.clip_R,
            spd_jitter=args.spd_jitter,
            seed=args.seed
        )
    elif (args.dp_mu_std > 0.0) or (args.dp_cov_eps > 0.0) or (args.dp_w_std > 0.0):
        print(f"[DP-like] Apply noise to GMM: mu_std={args.dp_mu_std}, cov_eps={args.dp_cov_eps}, w_std={args.dp_w_std}")
        rng_dp = np.random.RandomState(args.seed)
        per_label_enc = apply_noise_to_gmm(
            per_label_enc,
            mu_std=args.dp_mu_std,
            cov_eps=args.dp_cov_eps,
            w_std=args.dp_w_std,
            rng=rng_dp,
            mu_mode=args.dp_mu_mode,
            mu_coeff=args.dp_mu_coeff,
            inflate_cov_s=args.dp_inflate_cov_s
        )
    else:
        print("[DP] No DP noise (dp_mode=none and all DP-like params are 0).")

    targets = build_targets_from_gmm(per_label_enc,
                                     mode=args.mode,
                                     per_class=args.per_class,
                                     seed=args.seed)
    print(f"[INFO] targets: {len(targets)} (labels={sorted(set(l for l,_,_ in targets))})")

    print(f"[LOAD] PKL: {args.pkl}")
    base_ds = torch.load(args.pkl, weights_only=False)
    raw_ds  = RawDataset(base_ds, to_size=args.resize)
    Zdb, Ydb = encode_dataset_for_eval(base_ds, encoder,
                                       resize_to=args.resize,
                                       device=device,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers)
    if args.dp_mode == "gaussian_mech" and args.clip_R > 0:
        Zdb = l2_clip_(Zdb, args.clip_R)
    print(f"[INFO] Encoded DB: Z={tuple(Zdb.shape)}, labels={sorted(set(Ydb.numpy().tolist()))}")

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["idx","label","tag","cosine_nn","nn_idx","mse_pix","psnr","ssim","rec_path","pair_path"])

        cos_list, mse_list, psnr_list, ssim_list = [], [], [], []

        for idx, (lbl, zt, tag) in enumerate(targets):
            print(f"\n[ATTACK] {idx+1}/{len(targets)} label={lbl} tag={tag}")
            img_rec = feature_inversion(encoder, zt,
                                        steps=args.steps, lr=args.lr,
                                        tv_weight=args.tv, l2_weight=args.l2,
                                        init=args.init, device=device.type,
                                        out_size=args.resize, log_every=max(50, args.steps//10))

            with torch.no_grad():
                z_rec = encode_batch(inv_preprocess(img_rec.unsqueeze(0).to(device), args.resize), encoder).squeeze(0).cpu()
            if args.dp_mode == "gaussian_mech" and args.clip_R > 0:
                z_rec = l2_clip_(z_rec.unsqueeze(0), args.clip_R).squeeze(0)

            cos, nn_idx = cosine_top1(z_rec, Zdb)
            img_nn, _ = raw_ds[nn_idx]
            mse, psnr = mse_psnr(img_rec, img_nn)
            ssim_val = ssim_tensor(img_rec, img_nn)

            cos_list.append(cos); mse_list.append(mse); psnr_list.append(psnr); ssim_list.append(ssim_val)

            out_img = os.path.join(args.out_dir, f"rec_label{lbl}_{tag}.png")
            out_pair = os.path.join(args.out_dir, f"pair_label{lbl}_{tag}.png")
            save_image(img_rec, out_img)
            pair = torch.stack([img_rec, img_nn], dim=0)
            save_image(pair, out_pair, nrow=2)

            wcsv.writerow([idx, lbl, tag, f"{cos:.6f}", nn_idx, f"{mse:.8f}", f"{psnr:.3f}",
                           (f"{ssim_val:.4f}" if HAS_SKIMAGE else "nan"), out_img, out_pair])

            print(f"  cosine_nn={cos:.4f}  nn_idx={nn_idx}  mse={mse:.6e}  psnr={psnr:.2f} dB  ssim={ssim_val if HAS_SKIMAGE else float('nan'):.4f}")
            print(f"  saved: {out_img}  |  {out_pair}")


    def _fmt_mean_std(vs):
        arr = np.array([x for x in vs if np.isfinite(x)], dtype=float)
        if arr.size == 0:
            return "nan", "nan"
        return f"{arr.mean():.6f}", f"{arr.std():.6f}"

    m_cos, s_cos = _fmt_mean_std(cos_list)
    m_mse, s_mse = _fmt_mean_std(mse_list)
    m_psn, s_psn = _fmt_mean_std(psnr_list)
    m_ssi, s_ssi = _fmt_mean_std(ssim_list) if HAS_SKIMAGE else ("nan", "nan")

    summary_path = os.path.join(args.out_dir, "metrics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Inversion Evaluation (mean ± std) ===\n")
        f.write(f"Cosine (feature, NN): {m_cos} ± {s_cos}\n")
        f.write(f"MSE   (pixel, NN):    {m_mse} ± {s_mse}\n")
        f.write(f"PSNR  (pixel, NN):    {m_psn} ± {s_psn} dB\n")
        f.write(f"SSIM  (pixel, NN):    {m_ssi} ± {s_ssi}\n")
        f.write("\n=== DP-like Noise Config (if used) ===\n")
        f.write(f"dp_mu_std={args.dp_mu_std}\n")
        f.write(f"dp_cov_eps={args.dp_cov_eps}\n")
        f.write(f"dp_w_std={args.dp_w_std}\n")
        f.write(f"dp_mu_mode={args.dp_mu_mode}\n")
        f.write(f"dp_mu_coeff={args.dp_mu_coeff}\n")
        f.write(f"dp_inflate_cov_s={args.dp_inflate_cov_s}\n")

        if dp_meta is not None:
            f.write("\n=== Formal (ε,δ)-DP Config ===\n")
            for k, v in dp_meta.items():
                f.write(f"{k}={v}\n")

    print("\n=== Summary (mean ± std) ===")
    print(f"Cosine: {m_cos} ± {s_cos}")
    print(f"MSE:    {m_mse} ± {s_mse}")
    print(f"PSNR:   {m_psn} ± {s_psn} dB")
    print(f"SSIM:   {m_ssi} ± {s_ssi}")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Images under: {args.out_dir}")

if __name__ == "__main__":
    main()
