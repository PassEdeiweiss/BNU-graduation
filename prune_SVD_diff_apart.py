import torch
import torch.nn as nn
import os
import re
from transformers import AutoModel, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration, VisionMlp
)
import matplotlib.pyplot as plt


def prune_module_with_separated_threshold(module, module_name, target_energy_ratio=0.99):
    if isinstance(module, VisionMlp):
        print(f"处理模块 {module_name}:")
        with torch.no_grad():
            # 获取原始权重
            fc1_weight = module.fc1.weight.data  # [out_dim, in_dim] = [hidden_dim, dim]
            fc2_weight = module.fc2.weight.data  # [out_dim, in_dim] = [dim, hidden_dim]

        # 对两个权重矩阵分别进行SVD分解
        U1, S1, Vh1 = torch.linalg.svd(fc1_weight, full_matrices=False)
        U2, S2, Vh2 = torch.linalg.svd(fc2_weight, full_matrices=False)

        # 计算保留的奇异值数量
        energy1 = (S1**2).cumsum(0) / (S1**2).sum()
        energy2 = (S2**2).cumsum(0) / (S2**2).sum()
        k1 = torch.nonzero(energy1 >= target_energy_ratio, as_tuple=False)[0].item() + 1
        k2 = torch.nonzero(energy2 >= target_energy_ratio, as_tuple=False)[0].item() + 1
        k = max(k1, k2)
        
        # 重建权重矩阵（确保形状正确）
        # fc1: [k, dim] = [hidden_dim_new, dim]
        new_fc1_weight = (Vh1[:k] * S1[:k].unsqueeze(1)).float()
        
        # fc2: [dim, k] = [dim, hidden_dim_new]
        new_fc2_weight = (U2[:, :k] * S2[:k].unsqueeze(0)).float()

        # 创建新的MLP模块
        pruned_mlp = VisionMlp(
            dim=module.fc1.in_features,
            hidden_dim=k,
            hidden_act="quick_gelu"
        )

        # 验证形状匹配
        assert new_fc1_weight.shape == pruned_mlp.fc1.weight.shape, \
            f"fc1形状不匹配: {new_fc1_weight.shape} vs {pruned_mlp.fc1.weight.shape}"
        assert new_fc2_weight.shape == pruned_mlp.fc2.weight.shape, \
            f"fc2形状不匹配: {new_fc2_weight.shape} vs {pruned_mlp.fc2.weight.shape}"

        # 赋值权重和偏置
        pruned_mlp.fc1.weight.data = new_fc1_weight
        pruned_mlp.fc2.weight.data = new_fc2_weight
        pruned_mlp.fc1.bias.data = module.fc1.bias[:k].clone()
        pruned_mlp.fc2.bias.data = module.fc2.bias.clone()

        print(f"层 {module_name} 剪枝至中间维度: {k}")
        return pruned_mlp, k

    return module, None


def print_model_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total:,}")
    return total


def get_layer_idx_from_name(name):
    match = re.search(r'visual\.blocks\.(\d+)\.mlp', name)
    return int(match.group(1)) if match else None


def prune_MLP_model_with_separated_threshold(model: nn.Module, target_energy_ratio=0.99):
    pruned_sizes = []
    original_sizes = model.config.vision_config.mlp_size
    new_sizes = original_sizes.copy()

    for name, module in model.named_modules():
        if isinstance(module, VisionMlp):
            idx = get_layer_idx_from_name(name)
            pruned_module, pruned_size = prune_module_with_separated_threshold(module, name, target_energy_ratio=0.99)
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
        
        # 处理fc1参数
        fc1_weight = state_dict[f"{prefix}.fc1.weight"]
        if fc1_weight.shape[0] > new_size:
            new_state_dict[f"{prefix}.fc1.weight"] = fc1_weight[:new_size]  # [new_size, dim]
            new_state_dict[f"{prefix}.fc1.bias"] = state_dict[f"{prefix}.fc1.bias"][:new_size]
        
        # 处理fc2参数（需要转置处理）
        fc2_weight = state_dict[f"{prefix}.fc2.weight"]
        if fc2_weight.shape[1] > new_size:
            new_state_dict[f"{prefix}.fc2.weight"] = fc2_weight[:, :new_size]  # [dim, new_size]
        
        print(f"调整层 {i}: {fc1_weight.shape} -> {new_size}")
    
    return new_state_dict

if __name__ == "__main__":
    # 设置模型路径和保存路径
    model_path = "./Qwen-2B-1"
    save_dir = "./saver/pruned_vision_model-Qwen-threshold-pruning-2"
    # target_energy_ratio = 1.00  # 能量阈值
    target_energy_ratio = 0.99

    # 加载模型和处理器
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    # # 执行剪枝和奇异值分析
    pruned_model, new_mlp_sizes = prune_MLP_model_with_separated_threshold(
        model, target_energy_ratio
    )

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # pruned_model = model
    # new_mlp_sizes = model.config.vision_config.mlp_size

    # 保存剪枝后的模型和配置
    pruned_model.config.to_json_file(os.path.join(save_dir, "config.json"))
    pruned_model.save_pretrained(save_dir)
    print(pruned_model)
    processor.save_pretrained(save_dir, use_fast=False)

    

    # 修剪模型的 state_dict 并保存
    original_state_dict = pruned_model.state_dict()
    pruned_state_dict = prune_state_dict_by_mlp_sizes(original_state_dict, new_mlp_sizes)
    torch.save(pruned_state_dict, os.path.join(save_dir, "pytorch_model_pruned.bin"))

    print(f"\n模型与裁剪后的权重已保存至 {save_dir}")
