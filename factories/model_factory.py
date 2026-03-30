import os, torch
from transformers import (
    AutoTokenizer, AutoImageProcessor,
    AutoModelForSequenceClassification, AutoModelForCausalLM,
    ViTForImageClassification, CLIPModel, CLIPProcessor, CLIPVisionModel
)
import torch.nn as nn

def _src(cfg):

    return cfg["model"].get("local_path") or cfg["model"]["name"]

def _maybe_apply_lora(model, cfg, is_causal_lm=False):
    l = cfg["model"].get("lora", {})
    if not l.get("enable", False):
        return model
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        print("[WARN] PEFT not installed, proceed without LoRA:", e); return model
    task = ( __import__("peft").TaskType.CAUSAL_LM ) if is_causal_lm else ( __import__("peft").TaskType.SEQ_CLS )
    lora = __import__("peft").LoraConfig(
        r=l.get("rank", 8),
        lora_alpha=l.get("alpha", 32),
        lora_dropout=l.get("dropout", 0.05),
        target_modules=l.get("target_modules", None),
        task_type=task,
        bias="none",
    )
    model = __import__("peft").get_peft_model(model, lora)
    return model

def build_model_and_processor(cfg):
    name = cfg["model"]["name"].lower()
    path = cfg["model"].get("local_path") or cfg["model"]["name"]
    task = cfg["task"]


    if name in ["vit-b/16","vit_b_16","vit-b16","vit-b-16"]:
        # 视觉分类
        proc = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        model = ViTForImageClassification.from_pretrained(path, num_labels=cfg["model"]["num_labels"], local_files_only=True)
        return _maybe_apply_lora(model, cfg, is_causal_lm=False), proc

    if name in ["clip-vit-large-patch14-336","clip-vit-l/14@336","clip-vit-l-14-336","clip"]:
        # 用 CLIP 的视觉塔做分类（冻结文本塔）
        clip_proc = CLIPProcessor.from_pretrained(path, local_files_only=True)
        try:
            vis = CLIPVisionModel.from_pretrained(path, local_files_only=True)
        except:
            # 如果只有 CLIPModel 目录
            vis = CLIPModel.from_pretrained(path, local_files_only=True).vision_model
        # 构建一个简单分类头
        class ClipVisionCls(nn.Module):
            def __init__(self, vision, num_labels):
                super().__init__()
                self.vision = vision
                hd = vision.config.hidden_size
                self.head = nn.Linear(hd, cfg["model"]["num_labels"])
            def forward(self, pixel_values=None, labels=None, **kwargs):
                out = self.vision(pixel_values=pixel_values)
                pooled = out.pooler_output  # [B, hidden]
                logits = self.head(pooled)
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                return type("Out", (), {"logits": logits, "loss": loss})
        model = ClipVisionCls(vis, cfg["model"]["num_labels"])
        return _maybe_apply_lora(model, cfg, is_causal_lm=False), clip_proc

    if name in ["llava-1.5-3b","llava","llava-1_5-3b"]:
        raise NotImplementedError("LLaVA 构建需要多模态对话数据与专用 collator，建议单独入口。")

    if task == "text-cls":
        tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=cfg["model"]["num_labels"], local_files_only=True)
        model = _maybe_apply_lora(model, cfg, is_causal_lm=False)
        return model, tok

    elif task == "causal-lm":
        tok = AutoTokenizer.from_pretrained(path, use_fast=True, padding_side="right", local_files_only=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
        model = _maybe_apply_lora(model, cfg, is_causal_lm=True)
        return model, tok

    elif task == "vision-cls":
        proc = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        model = ViTForImageClassification.from_pretrained(path, num_labels=cfg["model"]["num_labels"], local_files_only=True)
        model = _maybe_apply_lora(model, cfg, is_causal_lm=False)
        return model, proc

    else:
        raise ValueError(f"Unknown task: {task}")
