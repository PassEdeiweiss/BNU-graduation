import torch
import torch.nn as nn
import os
import re
from transformers import AutoModel, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration, VisionMlp
)
import matplotlib.pyplot as plt


def prune_module_with_eff_svd(module, module_name, target_energy_ratio=0.99):
    if isinstance(module, VisionMlp):
        print(f"Eff-SVD 剪枝模块 {module_name}:")

        with torch.no_grad():
            fc1_weight = module.fc1.weight.data  # [hidden_dim, dim]
            fc2_weight = module.fc2.weight.data  # [dim, hidden_dim]

            # 构造有效矩阵 eff = fc1 @ fc2 (5120 x 1280 @ 1280 x 5120 = 5120 x 5120)
            eff = fc1_weight @ fc2_weight  # [hidden_dim, hidden_dim]

            # 对 eff 做 SVD
            U, S, Vh = torch.linalg.svd(eff, full_matrices=False)

            # 计算保留能量比例
            energy = (S**2).cumsum(0) / (S**2).sum()
            k = torch.nonzero(energy >= target_energy_ratio, as_tuple=False)[0].item() + 1

            print(f"保留奇异值数 k={k}, 总能量占比={energy[k-1].item():.4f}")

            # 构造新的低秩表示
            # 1. eff ≈ U_k @ S_k @ Vh_k
            U_k = U[:, :k]                      # [hidden_dim, k]
            S_k = S[:k]                         # [k]
            Vh_k = Vh[:k, :]                    # [k, hidden_dim]

            # 2. 分解为 fc1_new 和 fc2_new
            # fc1_new: [k, dim] = Vh_k @ fc1          (推导见备注)
            # fc2_new: [dim, k] = fc2 @ U_k @ diag(S_k)^0.5

            # 注意：U_k, Vh_k 本身是 unitary，因此乘 sqrt(S) 更稳健
            sqrt_S = torch.sqrt(S_k)

            fc1_new = (Vh_k @ fc1_weight) * sqrt_S.view(-1, 1)   # [k, dim]
            fc2_new = (fc2_weight @ U_k) * sqrt_S.view(1, -1)    # [dim, k]

            # 创建新的 VisionMlp 模块
            pruned_mlp = VisionMlp(
                dim=module.fc1.in_features,
                hidden_dim=k,
                hidden_act="quick_gelu"
            )

            # 替换权重
            pruned_mlp.fc1.weight.data = fc1_new.float()
            pruned_mlp.fc1.bias.data = module.fc1.bias[:k].clone()
            pruned_mlp.fc2.weight.data = fc2_new.float()
            pruned_mlp.fc2.bias.data = module.fc2.bias.clone()

            print(f"→ 剪枝后中间维度: {k} / 原始: {module.fc1.weight.size(0)}")
            return pruned_mlp, k

    return module, None


def print_model_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total:,}")
    return total


def get_layer_idx_from_name(name):
    match = re.search(r'visual\.blocks\.(\d+)\.mlp', name)
    return int(match.group(1)) if match else None


def prune_MLP_model_with_eff_svd(model: nn.Module, target_energy_ratio=0.99):
    pruned_sizes = []
    original_sizes = model.config.vision_config.mlp_size
    new_sizes = original_sizes.copy()

    for name, module in model.named_modules():
        if isinstance(module, VisionMlp):
            idx = get_layer_idx_from_name(name)
            pruned_module, pruned_size = prune_module_with_eff_svd(module, name, target_energy_ratio)
            if idx is not None:
                new_sizes[idx] = pruned_size
                parent_name = name.rsplit('.', 1)[0]
                attr_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, attr_name, pruned_module)
                pruned_sizes.append(pruned_size)

    model.config.vision_config.mlp_size = new_sizes
    print("更新后的 mlp_size:", new_sizes)
    return model, new_sizes


def prune_state_dict_by_mlp_sizes(state_dict: dict, new_mlp_sizes: list):
    new_state_dict = state_dict.copy()
    
    for i, new_size in enumerate(new_mlp_sizes):
        prefix = f"visual.blocks.{i}.mlp"
        
        # fc1
        fc1_weight = state_dict[f"{prefix}.fc1.weight"]
        if fc1_weight.shape[0] > new_size:
            new_state_dict[f"{prefix}.fc1.weight"] = fc1_weight[:new_size]
            new_state_dict[f"{prefix}.fc1.bias"] = state_dict[f"{prefix}.fc1.bias"][:new_size]
        
        # fc2
        fc2_weight = state_dict[f"{prefix}.fc2.weight"]
        if fc2_weight.shape[1] > new_size:
            new_state_dict[f"{prefix}.fc2.weight"] = fc2_weight[:, :new_size]
        
        print(f"调整层 {i}: {fc1_weight.shape} -> {new_size}")
    
    return new_state_dict


if __name__ == "__main__":
    model_path = "./Qwen-2B-1"
    save_dir = "./saver/pruned_vision_model-eff-svd"
    target_energy_ratio = 0.9999

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    pruned_model, new_mlp_sizes = prune_MLP_model_with_eff_svd(
        model, target_energy_ratio
    )

    os.makedirs(save_dir, exist_ok=True)

    pruned_model.config.to_json_file(os.path.join(save_dir, "config.json"))
    pruned_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir, use_fast=False)

    original_state_dict = pruned_model.state_dict()
    pruned_state_dict = prune_state_dict_by_mlp_sizes(original_state_dict, new_mlp_sizes)
    torch.save(pruned_state_dict, os.path.join(save_dir, "pytorch_model_pruned.bin"))

    print(f"\n模型与裁剪后的权重已保存至 {save_dir}")
