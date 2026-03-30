# file: gmm_ot_batch_similarity.py
import os
import glob
import argparse
import numpy as np
import warnings
from typing import List, Dict, Any, Tuple, Optional

from scipy.linalg import eigh
import ot  # pip install pot


# ======================== 读取与预处理 ========================

def load_gmm_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    labels = data["labels_sorted"].astype(int).tolist()

    per_label = {}
    for lbl in labels:
        w = data[f"{lbl}_weights"]
        m = data[f"{lbl}_means"]
        C = data[f"{lbl}_covariances"]
        cov_type = str(data[f"{lbl}_covariance_type"][0])
        n = int(data[f"{lbl}_n"][0])
        per_label[lbl] = dict(weights=w, means=m, covs=C, cov_type=cov_type, n=n)

    # 可选 PCA 参数
    pca = None
    if "pca_components_" in data.files and "pca_mean_" in data.files:
        comps = data["pca_components_"]
        mean = data["pca_mean_"]
        whit = bool(data["pca_whiten"][0]) if "pca_whiten" in data.files else False
        var = data.get("pca_explained_variance_", None)
        pca = dict(components=comps, mean=mean, whiten=whit, explained_variance=var)

    total = max(1, sum(per_label[l]["n"] for l in labels))
    class_weights = np.array([per_label[l]["n"] / total for l in labels], dtype=float)

    meta = {k: data[k][0] for k in data.files if k.startswith("meta_")}
    return dict(labels=labels, per_label=per_label, class_weights=class_weights, pca=pca, meta=meta)


def inverse_pca_means_covs(means, covs, cov_type, pca):
    comps = pca["components"]          # (M, D)
    mean0 = pca["mean"]                # (D,)
    whiten = bool(pca.get("whiten", False))
    var = pca.get("explained_variance", None)  # (M,) or None

    M, D = comps.shape
    if whiten:
        if var is None:
            raise ValueError("PCA whiten=True 但缺少 explained_variance_")
        s = np.sqrt(var)
        A = (comps.T * s).T           # (M,D)
    else:
        A = comps

    means_x = means @ A + mean0  # (G, D)

    if cov_type == "diag":
        covs_z = np.array([np.diag(cov) for cov in covs], dtype=float)  # (G,M,M)
    elif cov_type in ("full",):
        covs_z = covs.astype(float)
    else:
        if covs.ndim == 3 and covs.shape[1] == M and covs.shape[2] == M:
            covs_z = covs.astype(float)
        else:
            raise ValueError(f"Unsupported cov_type during inverse PCA: {cov_type}, covs shape={covs.shape}")

    AT = A.T  # (D,M)
    covs_x = np.empty((covs_z.shape[0], D, D), dtype=float)
    for g in range(covs_z.shape[0]):
        covs_x[g] = AT @ covs_z[g] @ A
    return means_x, covs_x


def maybe_inverse_to_encoder_space(model):
    """
    返回：labels, {lbl: dict(weights, means(G,D), covs(G,D,D))}, class_weights
    """
    labels = model["labels"]
    per_label = model["per_label"]
    class_w = model["class_weights"]
    pca = model["pca"]

    out = {}
    for lbl in labels:
        w = per_label[lbl]["weights"].astype(float)
        m = per_label[lbl]["means"]
        C = per_label[lbl]["covs"]
        cov_type = per_label[lbl]["cov_type"]

        if pca is not None:
            m_x, C_x = inverse_pca_means_covs(m, C, cov_type, pca)
        else:
            if cov_type == "diag":
                C_x = np.array([np.diag(ci) for ci in C], dtype=float)
                m_x = m.astype(float)
            elif cov_type in ("full",):
                C_x = C.astype(float); m_x = m.astype(float)
            else:
                if C.ndim == 3 and C.shape[1] == m.shape[1]:
                    C_x = C.astype(float); m_x = m.astype(float)
                else:
                    raise ValueError(f"Unsupported cov_type: {cov_type}")

        out[lbl] = dict(weights=w, means=m_x, covs=C_x)

    return labels, out, class_w


# ======================== 可选：对 GMM 加噪 ========================

def bool_arg(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "y", "t")

def apply_noise_to_gmm(prepared: Dict[int, Dict[str, np.ndarray]],
                       mu_std: float = 0.0,
                       cov_eps: float = 0.0,
                       w_std: float = 0.0,
                       rng: Optional[np.random.RandomState] = None) -> Dict[int, Dict[str, np.ndarray]]:
    """
    对每个 label 的 GMM 参数加噪：
      - means: 加 N(0, mu_std^2 I)
      - covs:  加 cov_eps * I  （保持 PSD）
      - weights: 加 N(0, w_std^2)，再截断为正并 L1 归一
    """
    if rng is None:
        rng = np.random.RandomState()

    out = {}
    for lbl, g in prepared.items():
        w = g["weights"].astype(float).copy()
        m = g["means"].astype(float).copy()
        C = g["covs"].astype(float).copy()

        G, D = m.shape

        # means noise
        if mu_std > 0:
            m += rng.normal(0.0, mu_std, size=m.shape)

        # cov noise (ensure PSD)
        if cov_eps > 0:
            C += np.eye(D)[None, ...] * cov_eps

        # weights noise
        if w_std > 0:
            w = w + rng.normal(0.0, w_std, size=w.shape)
            w = np.clip(w, 1e-12, None)
        # L1 normalize
        s = w.sum()
        w = w / (s if s > 0 else 1.0)

        out[lbl] = dict(weights=w, means=m, covs=C)

    return out


# ======================== 距离与 OT ========================

def _sym_sqrt(mat):
    vals, vecs = eigh(mat, check_finite=False)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def w2_gaussians(mu1, C1, mu2, C2, eps_jitter=1e-9):
    dmu2 = np.sum((mu1 - mu2) ** 2)
    C1s = C1 + np.eye(C1.shape[0]) * eps_jitter
    C2s = C2 + np.eye(C2.shape[0]) * eps_jitter
    C2_half = _sym_sqrt(C2s)
    middle = C2_half @ C1s @ C2_half
    middle_half = _sym_sqrt(middle)
    tr_term = np.trace(C1s + C2s - 2.0 * middle_half)
    return float(np.sqrt(max(dmu2 + tr_term, 0.0)))


def gmm_distance(gmmA, gmmB, reg=1e-2):
    wA, mA, CA = gmmA["weights"], gmmA["means"], gmmA["covs"]
    wB, mB, CB = gmmB["weights"], gmmB["means"], gmmB["covs"]

    GA, GB = len(wA), len(wB)
    Cmat = np.zeros((GA, GB), dtype=float)
    for a in range(GA):
        for b in range(GB):
            Cmat[a, b] = w2_gaussians(mA[a], CA[a], mB[b], CB[b])

    a = wA / max(1e-12, wA.sum())
    b = wB / max(1e-12, wB.sum())

    if reg is None or reg <= 0:
        pi = ot.emd(a, b, Cmat)
    else:
        reg_eff = float(reg) * (np.median(Cmat) + 1e-12)
        pi = ot.sinkhorn(a, b, Cmat, reg=reg_eff)

    dist = float((pi * Cmat).sum())
    return dist


def clients_distance_prepared(preA, preB, a, b, labels_A, labels_B,
                              inner_reg=1e-2, outer_reg=1e-2) -> float:
    Ki, Kj = len(labels_A), len(labels_B)
    GW = np.zeros((Ki, Kj), dtype=float)
    for i, li in enumerate(labels_A):
        for j, lj in enumerate(labels_B):
            GW[i, j] = gmm_distance(preA[li], preB[lj], reg=inner_reg)

    if outer_reg is None or outer_reg <= 0:
        gamma = ot.emd(a, b, GW)
    else:
        reg_eff = float(outer_reg) * (np.median(GW) + 1e-12)
        gamma = ot.sinkhorn(a, b, GW, reg=reg_eff)

    return float((gamma * GW).sum())


# ======================== 距离→相似度 + 行归一化 ========================

def distance_to_similarity(D: np.ndarray, mode: str = "exp", tau: float = 0.0) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if mode == "exp":
        if tau <= 0:
            nz = D[D > 0]
            tau = float(np.median(nz)) if nz.size > 0 else 1.0
        S = np.exp(-D / (tau + 1e-12))
    elif mode == "one_over":
        S = 1.0 / (1.0 + D)
    elif mode == "neg":
        S = -D
    else:
        raise ValueError(f"Unknown similarity mode: {mode}")
    return S


def row_normalize(M: np.ndarray, how: str = "minmax") -> np.ndarray:
    M = np.array(M, dtype=float, copy=True)
    if how == "minmax":
        mins = M.min(axis=1, keepdims=True)
        maxs = M.max(axis=1, keepdims=True)
        denom = np.clip(maxs - mins, 1e-12, None)
        return (M - mins) / denom
    elif how == "softmax":
        Z = M - M.max(axis=1, keepdims=True)
        np.exp(Z, out=Z)
        Z_sum = Z.sum(axis=1, keepdims=True) + 1e-12
        return Z / Z_sum
    elif how == "l1":
        s = np.abs(M).sum(axis=1, keepdims=True) + 1e-12
        return M / s
    elif how == "l2":
        s = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        return M / s
    else:
        raise ValueError(f"Unknown normalize method: {how}")


# ======================== 预加载（可选加噪） & 批量计算 ========================

def list_npz(folder: str, pattern: str = "*.npz") -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No npz files found in: {folder} (pattern={pattern})")
    return files


def preload_models(files: List[str],
                   noise_enable: bool,
                   noise_mu_std: float,
                   noise_cov_eps: float,
                   noise_w_std: float,
                   noise_seed: int):
    """一次性加载所有 npz -> 逆PCA到 encoder 空间 -> （可选）加噪 -> 缓存"""
    prepared = []
    rng = np.random.RandomState(noise_seed)
    for path in files:
        M = load_gmm_npz(path)
        labels, per_label, class_w = maybe_inverse_to_encoder_space(M)
        if noise_enable:
            # 每个客户端单独采样噪声（用同一个 rng 也 OK，因为我们只关心统计扰动）
            per_label = apply_noise_to_gmm(per_label,
                                           mu_std=noise_mu_std,
                                           cov_eps=noise_cov_eps,
                                           w_std=noise_w_std,
                                           rng=rng)
        prepared.append((labels, per_label, class_w))
    return prepared


def pairwise_distance_matrix(files: List[str],
                             prepared,
                             inner_reg: float,
                             outer_reg: float) -> np.ndarray:
    n = len(files)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i, i] = 0.0
        li, per_i, ai = prepared[i]
        for j in range(i + 1, n):
            lj, per_j, bj = prepared[j]
            dij = clients_distance_prepared(per_i, per_j, ai, bj, li, lj,
                                            inner_reg=inner_reg, outer_reg=outer_reg)
            D[i, j] = D[j, i] = dij
            print(f"[{i},{j}] distance = {dij:.6f}")
    return D


def save_matrix_csv(path: str, M: np.ndarray, header_labels: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(["file"] + header_labels) + "\n")
        for name, row in zip(header_labels, M):
            f.write(",".join([name] + [f"{v:.8f}" for v in row]) + "\n")


# ======================== CLI ========================

def main():
    ap = argparse.ArgumentParser(description="Batch compute client similarity matrix from GMM npz folder (optional DP-like noise).")
    ap.add_argument("--folder", type=str, required=True, help="Folder containing *.npz (each is a client's GMM)")
    ap.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for npz files")
    ap.add_argument("--inner_reg", type=float, default=1e-2, help="Sinkhorn reg for component-level OT")
    ap.add_argument("--outer_reg", type=float, default=1e-2, help="Sinkhorn reg for class-level OT")
    ap.add_argument("--sim_mode", type=str, default="exp", choices=["exp", "one_over", "neg"],
                    help="How to convert distance to similarity")
    ap.add_argument("--tau", type=float, default=0.0, help="Temperature for exp mode; <=0 -> auto (median)")
    ap.add_argument("--row_norm", type=str, default="l1", choices=["minmax", "softmax", "l1", "l2"],
                    help="Row-wise normalization method on the similarity matrix")
    ap.add_argument("--out_prefix", type=str, default="gmm_clients",
                    help="Prefix for output files (CSV and NPY)")

    # 噪声相关（布尔用可显式 True/False 形式）
    ap.add_argument("--noise_enable", type=bool_arg, default=False,
                    help="Enable adding noise to GMM params before distance (True/False)")
    ap.add_argument("--noise_mu_std", type=float, default=0.0,
                    help="Std of Gaussian noise added to means (encoder space), 0 disables")
    ap.add_argument("--noise_cov_eps", type=float, default=0.0,
                    help="Epsilon added to covariances as cov += eps * I, 0 disables")
    ap.add_argument("--noise_w_std", type=float, default=0.0,
                    help="Std of Gaussian noise added to weights before re-normalization, 0 disables")
    ap.add_argument("--noise_seed", type=int, default=2025, help="Random seed for noise sampler")

    args = ap.parse_args()

    files = list_npz(args.folder, args.pattern)
    names = [os.path.basename(p) for p in files]
    print(f"[INFO] Found {len(files)} npz files:")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")

    # 0) 预加载（逆PCA -> （可选）加噪）
    print("\n[STEP 0] Preloading models and applying noise:", args.noise_enable)
    prepared = preload_models(files,
                              noise_enable=args.noise_enable,
                              noise_mu_std=args.noise_mu_std,
                              noise_cov_eps=args.noise_cov_eps,
                              noise_w_std=args.noise_w_std,
                              noise_seed=args.noise_seed)

    # 1) 距离矩阵（越小越相似）
    print("\n[STEP 1] Computing pairwise distances ...")
    D = pairwise_distance_matrix(files, prepared, inner_reg=args.inner_reg, outer_reg=args.outer_reg)

    # 2) 距离 -> 相似度
    print("\n[STEP 2] Converting distance to similarity ...")
    S = distance_to_similarity(D, mode=args.sim_mode, tau=args.tau)

    # 3) 行归一化（默认 l1：每行和=1，适合作为聚合权重）
    print(f"\n[STEP 3] Row normalization: {args.row_norm}")
    S_row = row_normalize(S, how=args.row_norm)

    # 4) 打印尺寸与示例
    print("\n=== Shapes ===")
    print("Distance matrix D:", D.shape)
    print("Similarity matrix S:", S.shape)
    print("Row-normalized S_row:", S_row.shape)
    show = min(5, len(files))
    print("\nTop-left block of S_row:")
    print(np.round(S_row[:show, :show], 4))

    # 5) 保存
    np.save(f"{args.out_prefix}_distance.npy", D)
    np.save(f"{args.out_prefix}_similarity.npy", S)
    np.save(f"{args.out_prefix}_similarity_rownorm.npy", S_row)

    save_matrix_csv(f"{args.out_prefix}_distance.csv", D, names)
    save_matrix_csv(f"{args.out_prefix}_similarity.csv", S, names)
    save_matrix_csv(f"{args.out_prefix}_similarity_rownorm.csv", S_row, names)

    print(f"\n[SAVED] Matrices saved to:")
    print(f"  {args.out_prefix}_distance.npy / .csv")
    print(f"  {args.out_prefix}_similarity.npy / .csv")
    print(f"  {args.out_prefix}_similarity_rownorm.npy / .csv")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
