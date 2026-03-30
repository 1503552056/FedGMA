# file: gmm_ot_batch_similarity.py
import os
import glob
import argparse
import numpy as np
import warnings
from typing import List, Dict, Any, Tuple

from scipy.linalg import eigh
import ot  # pip install pot


# ======================== 从单对计算中复用的核心函数 ========================

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

    pca = None
    if "pca_components_" in data.files and "pca_mean_" in data.files:
        comps = data["pca_components_"]            # (M, D_orig)
        mean = data["pca_mean_"]                   # (D_orig,)
        whit = bool(data["pca_whiten"][0]) if "pca_whiten" in data.files else False
        var = data.get("pca_explained_variance_", None)  # (M,)
        pca = dict(components=comps, mean=mean, whiten=whit, explained_variance=var)

    total = sum(per_label[l]["n"] for l in labels)
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
        s = np.sqrt(var)              # (M,)
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
    labels = model["labels"]
    per_label = model["per_label"]
    class_w = model["class_weights"]
    pca = model["pca"]

    out = {}
    for lbl in labels:
        w = per_label[lbl]["weights"]
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

        out[lbl] = dict(weights=w.astype(float), means=m_x, covs=C_x)

    return labels, out, class_w


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

    a = wA / wA.sum()
    b = wB / wB.sum()

    if reg is None or reg <= 0:
        pi = ot.emd(a, b, Cmat)
    else:
        reg_eff = float(reg) * (np.median(Cmat) + 1e-12)
        pi = ot.sinkhorn(a, b, Cmat, reg=reg_eff)

    dist = float((pi * Cmat).sum())
    return dist


def clients_distance(npz_A, npz_B, inner_reg=1e-2, outer_reg=1e-2) -> float:
    A = load_gmm_npz(npz_A)
    B = load_gmm_npz(npz_B)

    labels_A, per_A, a = maybe_inverse_to_encoder_space(A)
    labels_B, per_B, b = maybe_inverse_to_encoder_space(B)

    Ki, Kj = len(labels_A), len(labels_B)
    GW = np.zeros((Ki, Kj), dtype=float)
    for i, li in enumerate(labels_A):
        for j, lj in enumerate(labels_B):
            GW[i, j] = gmm_distance(per_A[li], per_B[lj], reg=inner_reg)

    if outer_reg is None or outer_reg <= 0:
        gamma = ot.emd(a, b, GW)
    else:
        reg_eff = float(outer_reg) * (np.median(GW) + 1e-12)
        gamma = ot.sinkhorn(a, b, GW, reg=reg_eff)

    return float((gamma * GW).sum())  # S_ij (越小越相似)


# ======================== 距离→相似度 + 行归一化 ========================

def distance_to_similarity(D: np.ndarray, mode: str = "exp", tau: float = 0.0) -> np.ndarray:
    """
    将距离矩阵 D 转成相似度矩阵 S。
    mode:
      - "exp": S = exp(-D / tau)，tau<=0则用非零距离的中位数
      - "one_over": S = 1 / (1 + D)
      - "neg": S = -D（仅用于想保留相对顺序时）
    """
    D = np.asarray(D, dtype=float)
    if mode == "exp":
        if tau <= 0:
            # 用非零项的中位数作为尺度，避免tau难调
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
    """
    对每一行做归一化：
      - "minmax": (x - min) / (max - min)  -> [0,1]；若 max==min 则返回全0
      - "softmax": softmax(x)（注意这会让较大的值权重更大）
      - "l1": 每行除以行和（和为0则保持0）
      - "l2": 每行除以行范数（范数为0则保持0）
    """
    M = np.array(M, dtype=float, copy=True)

    if how == "minmax":
        mins = M.min(axis=1, keepdims=True)
        maxs = M.max(axis=1, keepdims=True)
        denom = np.clip(maxs - mins, 1e-12, None)
        return (M - mins) / denom
    elif how == "softmax":
        # 数值稳定 softmax
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


# ======================== 主流程：批量计算 ========================

def list_npz(folder: str, pattern: str = "*.npz") -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No npz files found in: {folder} (pattern={pattern})")
    return files


def pairwise_distance_matrix(files: List[str], inner_reg: float, outer_reg: float) -> np.ndarray:
    n = len(files)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            dij = clients_distance(files[i], files[j], inner_reg=inner_reg, outer_reg=outer_reg)
            D[i, j] = D[j, i] = dij
            print(f"[{i},{j}] distance = {dij:.6f}")
    return D


def save_matrix_csv(path: str, M: np.ndarray, header_labels: List[str]):
    # 友好的 CSV：第一列是文件名
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(["file"] + header_labels) + "\n")
        for name, row in zip(header_labels, M):
            f.write(",".join([name] + [f"{v:.8f}" for v in row]) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Batch compute client similarity matrix from GMM npz folder.")
    ap.add_argument("--folder", type=str, required=True, help="Folder containing *.npz (each is a client's GMM)")
    ap.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for npz files")
    ap.add_argument("--inner_reg", type=float, default=1e-2, help="Sinkhorn reg for component-level OT")
    ap.add_argument("--outer_reg", type=float, default=1e-2, help="Sinkhorn reg for class-level OT")
    ap.add_argument("--sim_mode", type=str, default="exp", choices=["exp", "one_over", "neg"],
                    help="How to convert distance to similarity")
    ap.add_argument("--tau", type=float, default=0.0, help="Temperature for exp mode; <=0 -> auto (median)")
    ap.add_argument("--row_norm", type=str, default="minmax", choices=["minmax", "softmax", "l1", "l2"],
                    help="Row-wise normalization method on the similarity matrix")
    ap.add_argument("--out_prefix", type=str, default="gmm_clients",
                    help="Prefix for output files (CSV and NPY)")
    args = ap.parse_args()

    files = list_npz(args.folder, args.pattern)
    names = [os.path.basename(p) for p in files]
    print(f"[INFO] Found {len(files)} npz files:")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")

    # 1) 距离矩阵（越小越相似）
    print("\n[STEP 1] Computing pairwise distances ...")
    D = pairwise_distance_matrix(files, inner_reg=args.inner_reg, outer_reg=args.outer_reg)

    # 2) 距离 -> 相似度
    print("\n[STEP 2] Converting distance to similarity ...")
    S = distance_to_similarity(D, mode=args.sim_mode, tau=args.tau)

    # 3) 行归一化
    print(f"\n[STEP 3] Row normalization: {args.row_norm}")
    S_row = row_normalize(S, how=args.row_norm)

    # 4) 打印头部/尺寸
    print("\n=== Shapes ===")
    print("Distance matrix D:", D.shape)
    print("Similarity matrix S:", S.shape)
    print("Row-normalized S_row:", S_row.shape)

    # 打印一个 5x5 左上角块作为示例
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
