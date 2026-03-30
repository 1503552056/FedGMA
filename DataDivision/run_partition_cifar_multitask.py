# file: run_partition_cifar_multitask_global_layout.py
import os
import json
import csv  # === NEW: 导出CSV ===
from collections import Counter

import torch
import torchvision
import torchvision.transforms as transforms

# ===== 任务定义 =====
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASSNAME2ID = {name: i for i, name in enumerate(CIFAR10_CLASSES)}

TASKS_ORDERED = [
    ("Transportation", ["airplane", "automobile", "ship", "truck"]),  # 4
    ("Mammals-Domestic-Herbivore", ["cat", "deer", "horse"]),         # 3
    ("Other-Animals", ["bird", "dog", "frog"]),                       # 3
]
TASK2CLASSIDS = {t: [CLASSNAME2ID[n] for n in names] for t, names in TASKS_ORDERED}
ID2NAME = {i: n for i, n in enumerate(CIFAR10_CLASSES)}

# ===== 文本描述（保持原风格） =====
def make_summary_text(task_name, label_percents, top_k=4):
    top = sorted(label_percents.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_labels = ", ".join([f"{k}" for k, _ in top])
    dist_str = ", ".join([f"{k} {int(v*100)}%" for k, v in top])
    return (
        f"This client contains CIFAR-10 images for the {task_name} task.\n"
        f"Main categories: {top_labels}.\n"
        f"Estimated label distribution: {dist_str}.\n"
        f"Images are 32x32 RGB natural photos with varied backgrounds and viewpoints."
    )

def percent_from_counts(cnt_by_name):
    total = sum(cnt_by_name.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in cnt_by_name.items()}


# ===== 载入扩展后的 PartitionCIFAR =====
from dataset_deal import PartitionCIFAR  # 需支持 label_whitelist 与 split-centric

if __name__ == "__main__":
    # 基础参数
    root_dir  = "./dataset/cifar_raw"
    save_root = "./dataset/cifar_partitioned_global_3x3_0.2"  # 目标目录（全局统一布局）
    os.makedirs(save_root, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(save_root, split), exist_ok=True)

    num_clients_per_task = 3
    partition = "dirichlet"
    dir_alpha = 0.2
    seed = 123
    val_ratio = 0.1
    test_like_train = True

    # 仅用于统计标签
    train_raw = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True)
    raw_labels = train_raw.targets if isinstance(train_raw.targets, list) else list(train_raw.targets)

    # 变换
    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    global_cid = 0
    descriptions = {}
    mapping = {}
    label_rows = []  # === NEW: 记录每个客户端的标签统计（train split）

    for task_name, class_ids in TASK2CLASSIDS.items():
        print(f"\n=== Task: {task_name} -> classes {class_ids} ===")

        tmp_task_path = os.path.join(save_root, f"__tmp__{task_name}")
        os.makedirs(tmp_task_path, exist_ok=True)

        pc = PartitionCIFAR(
            root=root_dir,
            path=tmp_task_path,
            dataname="cifar10",
            num_clients=num_clients_per_task,
            download=True,
            preprocess=True,
            balance=True,
            partition=partition,
            dir_alpha=dir_alpha,
            seed=seed,
            train_transform=train_transform,
            val_transform=eval_transform,
            test_transform=eval_transform,
            val_ratio=val_ratio,
            test_like_train=test_like_train,
            label_whitelist=class_ids,
            save_layout="split-centric",
        )

        for local_cid in range(num_clients_per_task):
            ds_train = pc.get_dataset(local_cid, type="train")
            ds_val   = pc.get_dataset(local_cid, type="val")
            ds_test  = pc.get_dataset(local_cid, type="test")

            # === 统计当前客户端的 train 标签分布 ===
            idxs = getattr(ds_train, "indices", None)
            if idxs is not None:
                label_list = [int(raw_labels[i]) for i in idxs]
            else:
                label_list = []
                for i in range(len(ds_train)):
                    _, y = ds_train[i]
                    y = int(y.item() if hasattr(y, "item") else y)
                    label_list.append(y)

            wl = set(class_ids)
            label_list = [l for l in label_list if l in wl]
            cnt = Counter(label_list)
            cnt_by_name = {ID2NAME[k]: int(v) for k, v in cnt.items()}
            perc = percent_from_counts(cnt_by_name)
            summary = make_summary_text(task_name, perc, top_k=4)

            # === NEW: 记录该客户端的标签分布行 ===
            row = {"client": f"client{global_cid}", "task": task_name, "split": "train"}
            for cname in CIFAR10_CLASSES:
                row[cname] = cnt_by_name.get(cname, 0)
            label_rows.append(row)

            # === 保存数据集 ===
            out_train = os.path.join(save_root, "train", f"client{global_cid}.pkl")
            out_val   = os.path.join(save_root, "val",   f"client{global_cid}.pkl")
            out_test  = os.path.join(save_root, "test",  f"client{global_cid}.pkl")

            torch.save(ds_train, out_train)
            torch.save(ds_val,   out_val)
            torch.save(ds_test,  out_test)

            descriptions[f"client{global_cid}"] = summary
            mapping[f"client{global_cid}"] = {
                "task": task_name,
                "local_cid": local_cid
            }

            print(f"Saved client{global_cid} from task {task_name} (local {local_cid})")
            global_cid += 1

        try:
            import shutil
            shutil.rmtree(tmp_task_path, ignore_errors=True)
        except Exception:
            pass

    # === 写出全局描述与映射 ===
    with open(os.path.join(save_root, "descriptions.json"), "w") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    with open(os.path.join(save_root, "mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    # === NEW: 写出标签分布 CSV ===
    csv_path = os.path.join(save_root, "label_distribution_train.csv")
    fieldnames = ["client", "task", "split"] + CIFAR10_CLASSES
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in label_rows:
            writer.writerow(r)
    print("Wrote label distribution CSV to:", csv_path)

    print("\nDone. Global layout saved to:", save_root)
    print("Total clients:", global_cid)
