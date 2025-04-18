import torch
import torch.nn as nn
import os
import copy
from copy import deepcopy
from transformers import AutoModel, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2MLP, Qwen2VLModel, VisionMlp, Qwen2VLForConditionalGeneration, VisionSdpaAttention
)
# torch.set_printoptions(profile="full")

def calculate_sparsity(weights, threshold=1e-5):
    """
    计算权重矩阵的稀疏性：接近零的元素的比例
    """
    return (weights.abs() < threshold).sum().item() / weights.numel()

def prune_attention_weights_by_sparsity(model, sparsity_threshold=0.5, max_prune_ratio=0.5):
    """
    基于稀疏性剪枝策略，剪掉指定的注意力权重。
    
    参数：
    model - 需要进行剪枝的模型
    sparsity_threshold - 稀疏性阈值，低于此值的模块将有更多的权重被剪枝
    max_prune_ratio - 最大剪枝比例，避免过度剪枝
    """
    pruned_model = copy.deepcopy(model)
    
    # 获取每个 VisionSdpaAttention 模块的稀疏性
    sparse_scores = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, VisionSdpaAttention):
            # 当前模块的 qkv 权重
            qkv_weights = module.qkv.weight.data
            sparsity = calculate_sparsity(qkv_weights)
            sparse_scores.append((name, sparsity))
    
    # 根据稀疏性排序，稀疏性高的模块排在前面
    sparse_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 为每个模块决定剪枝数量
    for name, sparsity in sparse_scores:
        module = pruned_model.get_submodule(name)
        
        qkv_weights = module.qkv.weight.data
        proj_weights = module.proj.weight.data
        
        num_attention_heads = module.qkv.weight.size(0) // 3  # qkv 是 [3 * num_attention_heads, embed_dim]
        max_weights_to_prune = num_attention_heads // 4  # 每个模块最多剪掉 4 个权重块
        
        # 计算剪枝的权重数量：如果稀疏性大于阈值，剪掉更多权重
        weights_to_prune_count = int(min(max_prune_ratio * num_attention_heads, max_weights_to_prune))
        
        print(f"Module {name} with sparsity {sparsity:.4f} will prune {weights_to_prune_count} weight blocks.")
        
        # 选择剪枝的权重块：根据模块的稀疏性，优先剪掉最不重要的权重
        for weight_index in range(weights_to_prune_count):
            start_idx = weight_index * proj_weights.shape[1] // num_attention_heads
            end_idx = (weight_index + 1) * proj_weights.shape[1] // num_attention_heads
            qkv_weights[weight_index] = 0  # 将 qkv 中的该权重部分置零
            proj_weights[:, start_idx:end_idx] = 0  # 将 proj 中的该权重块置零
        
        # 重新赋值到模块中
        module.qkv.weight.data = qkv_weights
        module.proj.weight.data = proj_weights
    
    return pruned_model



def generate_prune_mask_SVD(weight_matrix: torch.Tensor, prune_ratio: float, dim: int = 1) -> torch.Tensor:
    # if weight_matrix: 1280 * 5120
    # U: 1280 * 1280, Vh: 5120 * 5120, S: 1280
    # 默认输入是左乘，如，输入 X，则是 XW_1W_2
    U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)

    # 根据dim确定要保留的奇异值个数
    k = max(1, int(S.size(0) * (1 - prune_ratio))) # 5120 * (1 - prune_ratio)

    prune_mask = torch.zeros(Vh.size(0), dtype=torch.bool)
    prune_mask[:k] = True   # 把前 k 个元素设为 True

    return prune_mask


def prune_MLP_model(model: nn.Module, vision_prune_ratio: float = 0.25) -> tuple[nn.Module, dict]:
    pruned_intermediate_sizes = {}

    def prune_module(module, module_name, vision_prune_ratio = 0.25):
        if isinstance(module, VisionMlp):
            print(f"剪枝前的模块 {module_name}:")
            print(module)

            efficient_weight_fc2 = module.fc1.weight @ module.fc2.weight

            print(efficient_weight_fc2)

            # prune_mask 是一个 5120 长度的 bool 向量
            prune_mask = generate_prune_mask_SVD(
                weight_matrix=efficient_weight_fc2,
                prune_ratio=vision_prune_ratio,
                dim=1
            )
            pruned_intermediate_size_fc2 = prune_mask.sum().item()
            hidden_act = module.hidden_act if hasattr(module, 'hidden_act') else 'gelu'

            # 剪枝后的 MLP
            pruned_mlp = VisionMlp(
                dim=module.fc1.in_features,
                hidden_dim=pruned_intermediate_size_fc2,
                hidden_act=hidden_act
            )

            pruned_mlp.fc1.weight.data = module.fc1.weight[prune_mask, :].clone()
            pruned_mlp.fc2.weight.data = module.fc2.weight[:, prune_mask].clone()

            print("剪枝后的模块结构:")
            print(pruned_mlp)
            return pruned_mlp

        return module

    a = 0  # 进入 for 循环的总次数
    m = 0  # 找到子模块中的 VisionMlp 模块数
    for name, child in model.named_modules():
        if isinstance(child, VisionMlp):
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]] if '.' in name else model
            setattr(parent_module, name.split('.')[-1], prune_module(child, name))
            m = m + 1
        a = a + 1
    print("进入 for 循环的总次数为", a)
    print("找到子模块中的 VisionMlp 模块数为", m)

    return model, pruned_intermediate_sizes

def print_model_parameters(model: nn.Module):
    """打印模型参数信息"""
    print(f"模型 {model.__class__.__name__} 的参数：")
    for name, param in model.named_parameters():
        print(f"参数名称: {name}, 形状: {param.shape}, requires_grad: {param.requires_grad}")

def safe_prune_and_save(original_model, sparsity_threshold=0.5, max_prune_ratio=0.5, save_path="./saver/pruned_vision_model-SVD-head-25"):
    """
    安全剪枝并保存模型的函数（不修改原始模型），基于稀疏性
    """
    # 基于稀疏性剪枝
    pruned_model = prune_heads_by_sparsity(original_model, sparsity_threshold, max_prune_ratio)
    pruned_model.config.to_json_file(os.path.join(save_path, "config.json"))
    # 保存剪枝后的模型
    pruned_model.save_pretrained(save_path)
    
    # 还需要保存 AutoProcessor 配置
    processor = AutoProcessor.from_pretrained(original_model.config._name_or_path)
    processor.save_pretrained(save_path)
    
    print(f"Pruned model and processor saved to {save_path}")

    return pruned_model


if __name__ == "__main__":
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained("./Qwen-2B-1")
    processor = AutoProcessor.from_pretrained("./Qwen-2B-1")
    
    # print_model_parameters(model)
    
    # 执行剪枝
    vision_prune_ratio = 0.75  # 剪枝比例
    pruned_model, pruned_sizes = prune_MLP_model(model, vision_prune_ratio=vision_prune_ratio)

    pruned_model.config.vision_config.mlp_ratio = 3
    save_dir = "./saver/pruned_vision_model-SVD-head-25"
    os.makedirs(save_dir, exist_ok=True)

    print(pruned_model)
    print("完成基于 SVD 的 Vision MLP 剪枝。")
    pruned_pruned_model = safe_prune_and_save(pruned_model, sparsity_threshold=0.5, max_prune_ratio=0.5, save_path=save_dir)

    print(pruned_pruned_model)
    
    # 保存剪枝后的模型
    pruned_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir, use_fast=False)
    print("保存成功！")