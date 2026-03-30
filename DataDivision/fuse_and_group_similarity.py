# file: fuse_text_gmm_and_group.py
import os
import json
import argparse
import numpy as np

def norm01(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    mn, mx = float(A.min()), float(A.max())
    if mx - mn < 1e-12:
        return np.ones_like(A)
    return (A - mn) / (mx - mn)

def load_descriptions(desc_path: str) -> dict:
    with open(desc_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  # dict of key -> text

def get_summary_for_client(descriptions: dict, client_key: str) -> str:
    if client_key in descriptions:
        return descriptions[client_key]

    for k, v in descriptions.items():
        if k.endswith("/" + client_key):
            return v
    raise KeyError(f"Description for '{client_key}' not found in descriptions.json")

def infer_task_for_client(descriptions: dict, client_key: str) -> str:

    if client_key in descriptions:
        k = client_key
    else:
        k = None
        for kk in descriptions.keys():
            if kk.endswith("/" + client_key):
                k = kk
                break
    if k and "/" in k:
        return k.split("/")[0]
    return "default"

def compute_text_similarity(client_keys, descriptions, model_name="all-MiniLM-L6-v2",
                            cross_task_down=1.0) -> (np.ndarray, list):
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise SystemExit("请先安装: pip install sentence-transformers scikit-learn")

    texts = [get_summary_for_client(descriptions, ck) for ck in client_keys]
    tasks = [infer_task_for_client(descriptions, ck) for ck in client_keys]

    model = SentenceTransformer(model_name)
    E = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)   # (N, D)
    S = cosine_similarity(E, E)  # [-1,1]
    S = (S + 1.0) / 2.0          # -> [0,1]
    np.fill_diagonal(S, 1.0)

    if cross_task_down < 1.0:
        tasks_np = np.array(tasks)
        mask_cross = tasks_np[:, None] != tasks_np[None, :]
        S[mask_cross] *= float(cross_task_down)

    return S.astype(np.float32), tasks

def load_gmm_csv(csv_path: str):
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("请先安装: pip install pandas")
    df = pd.read_csv(csv_path, index_col=0)
    row_keys = df.index.tolist()
    col_keys = df.columns.tolist()
    if row_keys != col_keys:
        common = [k for k in row_keys if k in col_keys]
        if len(common) == 0:
            raise ValueError("CSV 行列标签完全不匹配，无法对齐。")
        df = df.loc[common, common]
        row_keys = col_keys = common
    A = df.values.astype(np.float32)
    if not np.allclose(A, A.T, atol=1e-6):
        A = (A + A.T) / 2.0
    A = norm01(A)
    np.fill_diagonal(A, 1.0)
    return A, row_keys

def strip_suffix(keys):
    # client0.npz -> client0
    out = []
    for k in keys:
        base = os.path.basename(k)
        if base.endswith(".npz"):
            base = base[:-4]
        out.append(base)
    return out

def group_by_similarity(S: np.ndarray, mode: str, n_groups: int = None, threshold: float = 0.65) -> np.ndarray:
    N = S.shape[0]
    if mode == "spectral":
        if n_groups is None or n_groups < 2:
            raise ValueError("mode=spectral 时必须提供 --n_groups 且 >=2")
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            raise SystemExit("请先安装: pip install scikit-learn")
        model = SpectralClustering(
            n_clusters=n_groups, affinity="precomputed",
            assign_labels="kmeans", random_state=42
        )
        labels = model.fit_predict(S)
        return labels.astype(np.int32)

    elif mode == "threshold":
        A = (S >= threshold).astype(np.uint8)
        visited = np.zeros(N, dtype=bool)
        labels = -np.ones(N, dtype=np.int32)
        gid = 0
        for i in range(N):
            if visited[i]:
                continue
            # BFS
            q = [i]
            visited[i] = True
            labels[i] = gid
            while q:
                u = q.pop()
                nbrs = np.where(A[u] > 0)[0]
                for v in nbrs:
                    if not visited[v]:
                        visited[v] = True
                        labels[v] = gid
                        q.append(v)
            gid += 1
        return labels
    else:
        raise ValueError(f"Unknown mode={mode}")

def main():
    ap = argparse.ArgumentParser(description="Fuse text & GMM similarities and group clients.")
    ap.add_argument("--gmm_csv", type=str, required=True, help="GMM 相似度 CSV 路径")
    ap.add_argument("--descriptions", type=str, required=True, help="descriptions.json 路径")
    ap.add_argument("--out_dir", type=str, default="./_fused_out", help="输出目录")

    ap.add_argument("--w_text", type=float, default=0.5, help="文本相似度权重")
    ap.add_argument("--w_gmm", type=float, default=0.5, help="GMM 相似度权重")
    ap.add_argument("--text_model", type=str, default="all-MiniLM-L6-v2", help="sentence-transformers 模型名")
    ap.add_argument("--cross_task_down", type=float, default=1.0, help="跨任务文本相似度降权系数∈(0,1]")

    ap.add_argument("--mode", type=str, default="spectral", choices=["spectral", "threshold"], help="分组方式")
    ap.add_argument("--n_groups", type=int, default=None, help="谱聚类簇数（mode=spectral 使用）")
    ap.add_argument("--threshold", type=float, default=0.65, help="阈值联通阈值（mode=threshold 使用）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)


    S_gmm, csv_keys = load_gmm_csv(args.gmm_csv)
    client_keys = strip_suffix(csv_keys)  
    N = len(client_keys)
    print(f"[GMM] {S_gmm.shape}, clients: {client_keys}")


    descriptions = load_descriptions(args.descriptions)
    S_text, tasks = compute_text_similarity(client_keys, descriptions,
                                            model_name=args.text_model,
                                            cross_task_down=args.cross_task_down)
    print(f"[TEXT] {S_text.shape} (model={args.text_model}), tasks: {tasks}")


    S_text_n = norm01(S_text)
    S_gmm_n  = norm01(S_gmm)
    S_fused  = args.w_text * S_text_n + args.w_gmm * S_gmm_n
    np.fill_diagonal(S_fused, 1.0)


    labels = group_by_similarity(S_fused, mode=args.mode,
                                 n_groups=args.n_groups, threshold=args.threshold)


    np.save(os.path.join(args.out_dir, "sim_text.npy"),  S_text.astype(np.float32))
    np.save(os.path.join(args.out_dir, "sim_gmm.npy"),   S_gmm.astype(np.float32))
    np.save(os.path.join(args.out_dir, "sim_fused.npy"), S_fused.astype(np.float32))

    with open(os.path.join(args.out_dir, "clients_order.json"), "w", encoding="utf-8") as f:
        json.dump(client_keys, f, indent=2, ensure_ascii=False)

    groups_json = []
    for i, g in enumerate(labels):
        groups_json.append({
            "idx": i,
            "client": client_keys[i],
            "task": tasks[i],
            "group": int(g)
        })
    with open(os.path.join(args.out_dir, "groups.json"), "w", encoding="utf-8") as f:
        json.dump(groups_json, f, indent=2, ensure_ascii=False)


    try:
        import pandas as pd
        pd.DataFrame(groups_json).to_csv(os.path.join(args.out_dir, "groups.csv"), index=False)
    except Exception:
        pass


    print("\n=== Group Summary ===")
    from collections import defaultdict
    bucket = defaultdict(list)
    for i, g in enumerate(labels):
        bucket[int(g)].append(client_keys[i])
    for g, members in bucket.items():
        print(f"Group {g}: {members}")

    def stats(M, name):
        return f"{name}: min={M.min():.4f}, max={M.max():.4f}, diag_mean={np.diag(M).mean():.4f}"
    print("\n" + stats(S_text, "S_text"))
    print(stats(S_gmm,  "S_gmm"))
    print(stats(S_fused,"S_fused"))
    print(f"\nSaved to: {args.out_dir}")
    print("Files: sim_text.npy, sim_gmm.npy, sim_fused.npy, clients_order.json, groups.json/groups.csv")

if __name__ == "__main__":
    main()
