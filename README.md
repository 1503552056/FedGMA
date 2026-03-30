# FedGMA: Federated Group-wise Meta-Aggregation for Fine-tuningLarge Foundation Models


## Abstract

Federated LoRA (F-LoRA) learning represents an important direction for fine-tuning large foundation models in a parameter-efficient manner, allowing multiple clients to collaborate without compromising local data privacy. Model aggregation on the server is a critical step, where existing F-LoRA algorithms are limited when facing strong task and data heterogeneity.
We propose Federated Group-wise Meta-Aggregation (FedGMA), a learning-based model aggregation framework for learning of F-LoRA. FedGMA operates by first constructing  coherent client groups using text-based task embeddings and generative data distribution models like GMM. Within these homogeneous groups, a Reptile-inspired meta-learning algorithm learns to aggregate LoRA experts via an MMoE network. Extensive experiments on diverse vision and language benchmarks demonstrate that FedGMA significantly outperforms existing F-LoRA baselines, enhancing model adaptability and personalization in cross-silo federated learning. We provide the core implementation of FedGMA in the supplementary material.

## Folder Structure
```grapha  
FedGMA/
├── server/          # Server-side logic (aggregation, meta-learning, handlers)
├── client/          # Client training logic, proxy batches, optimizers
├── data_specs/      # ClientSpec definitions, dataset grouping, groups.csv loader
├── factories/       # Model builders, processor builders, dataloader factories
├── configs/         # YAML configurations for training & experiments
├── DataDivision/    # Dataset preprocessing & partitioning scripts
├── run/             # Entry points and runnable scripts (e.g., run_offline.sh)
└── client_specs/    # Generated ClientSpec JSON files
```  




