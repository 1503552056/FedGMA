
import os, json, random, glob
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class ClientSpec:
    cid: int
    dataset: str                
    category: str               
    category_id: int
    shard_id: int               
    samples: List[str]         

def _list_subdirs(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def build_image_specs(dataset_name: str, root: str, categories: List[str], shards_per_category: int, seed: int) -> List[ClientSpec]:
    import glob
    rng = random.Random(seed)
    cats = categories if categories else _list_subdirs(root)
    specs: List[ClientSpec] = []
    cid = 0
    for cidx, cat in enumerate(cats):
        img_dir = os.path.join(root, cat)
        files = []
        for ext in ["*.jpg","*.jpeg","*.png","*.bmp","*.webp"]:
            files += glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
        files = [os.path.relpath(p, root) for p in sorted(files)]
        if len(files) == 0:
            print(f"[WARN] No images under {img_dir}")
            continue
        rng.shuffle(files)
        K = shards_per_category
        n = len(files)
        for k in range(K):
            beg = (k * n) // K
            end = ((k + 1) * n) // K
            shard = files[beg:end]
            specs.append(ClientSpec(cid=cid, dataset=dataset_name, category=cat, category_id=cidx, shard_id=k, samples=shard))
            cid += 1
    return specs

def build_amazon_specs(root: str, domains: List[str], shards_per_category: int, seed: int) -> List[ClientSpec]:
    rng = random.Random(seed)
    specs: List[ClientSpec] = []
    cid = 0
    for cidx, dom in enumerate(domains):
        droot = os.path.join(root, dom)
        train_csv = os.path.join(droot, "train.csv")
        assert os.path.exists(train_csv), f"Missing {train_csv}"
        n = sum(1 for _ in open(train_csv, "r", encoding="utf-8")) - 1
        idxs = list(range(n))
        rng.shuffle(idxs)
        K = shards_per_category
        for k in range(K):
            beg = (k * n) // K
            end = ((k + 1) * n) // K
            shard = [f"{train_csv}#{i}" for i in idxs[beg:end]]
            specs.append(ClientSpec(cid=cid, dataset="amazon_md", category=dom, category_id=cidx, shard_id=k, samples=shard))
            cid += 1
    return specs

def build_domainnet_hf_specs(root: str, shards_per_category: int, seed: int) -> List[ClientSpec]:
    import pandas as pd, glob, os
    data_dir = os.path.join(root, "data")
    assert os.path.isdir(data_dir), f"[DomainNet-HF] Not found: {data_dir}"
    train_parts = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    assert len(train_parts) > 0, f"[DomainNet-HF] No train-*.parquet in {data_dir}"

    domain_names = {0:"clipart", 1:"infograph", 2:"painting", 3:"quickdraw", 4:"real", 5:"sketch"}

    by_domain = {i: [] for i in range(6)}
    for pq_abs in train_parts:
        pq_rel = os.path.relpath(pq_abs, root)
        df = pd.read_parquet(pq_abs, columns=["domain", "label"])  
        for ridx, dom in enumerate(df["domain"].astype(int).tolist()):
            by_domain[int(dom)].append(f"{pq_rel}#{ridx}")

    rng = random.Random(seed)
    specs: List[ClientSpec] = []
    cid = 0
    K = shards_per_category
    for dom_id in range(6):
        lst = by_domain[dom_id]
        if len(lst) == 0:
            continue
        rng.shuffle(lst)
        n = len(lst)
        for k in range(K):
            beg = (k * n) // K
            end = ((k + 1) * n) // K
            shard = lst[beg:end]
            specs.append(ClientSpec(
                cid=cid,
                dataset="domainnet_hf",
                category=domain_names[dom_id],
                category_id=dom_id,
                shard_id=k,
                samples=shard
            ))
            cid += 1
    return specs

def build_pkl_partition_specs(root: str,
                              groups_csv: str,
                              dataset_name: str) -> List[ClientSpec]:
    
    specs: List[ClientSpec] = []

    if not os.path.exists(groups_csv):
        raise FileNotFoundError(f"[build_pkl_partition_specs] groups.csv not found: {groups_csv}")

    with open(groups_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            client_str = row["client"]      # e.g. "client0"
            cid = int(client_str.replace("client", ""))
            task = row.get("task", "default")
            group = int(row["group"])

            specs.append(
                ClientSpec(
                    cid=cid,
                    dataset=dataset_name,
                    category=task,         
                    category_id=group,     
                    shard_id=0,
                    samples=[]             
                )
            )

    return specs



def build_and_save_specs(cfg: Dict[str, Any]) -> List[ClientSpec]:
    ds = cfg["dataset"]["name"].lower()
    k  = int(cfg["federation"]["shards_per_category"])
    seed = int(cfg["federation"]["seed"])
    save_path = cfg["federation"]["save_spec_path"]

    if ds in ["domainnet", "officehome", "tinyimagenet"]:
        root = cfg["dataset"]["root"]
        cats = cfg["dataset"].get("categories", [])
        specs = build_image_specs(ds, root, cats, k, seed)
    elif ds == "domainnet_hf":
        root = cfg["dataset"]["root"]
        specs = build_domainnet_hf_specs(root, k, seed)
    elif ds == "amazon_md":
        root = cfg["dataset"]["amazon"]["root"]
        domains = cfg["dataset"]["amazon"]["domains"]
        specs = build_amazon_specs(root, domains, k, seed)
    elif ds == "pkl_partition":
        root = cfg["dataset"]["root"]
        groups_csv = cfg["dataset"].get("groups_csv", os.path.join(root, "groups.csv"))
        specs = build_pkl_partition_specs(root, groups_csv, dataset_name=ds)
    else:
        raise ValueError(ds)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in specs], f, ensure_ascii=False, indent=2)
    return specs

def load_specs(path: str) -> List[ClientSpec]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return [ClientSpec(**x) for x in arr]
