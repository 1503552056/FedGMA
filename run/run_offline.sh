
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/hf_cache
export PYTHONPATH=/data/liyongle/CVPR/FedMMOE:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7              
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m run.entry_mmoe_meta --config configs/train_task.yaml
