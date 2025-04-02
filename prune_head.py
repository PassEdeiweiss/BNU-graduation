import os
import copy
import torch
from transformers import AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, VisionSdpaAttention

def calculate_sparsity(weights, threshold=1e-5):
    """
    计算权重矩阵的稀疏性：接近零的元素的比例
    """
    return (weights.abs() < threshold).sum().item() / weights.numel()

def prune_heads_by_sparsity(model, sparsity_threshold=0.5, max_prune_ratio=0.5):
    """
    基于稀疏性剪枝策略，剪掉指定的头。
    
    参数：
    model - 需要进行剪枝的模型
    sparsity_threshold - 稀疏性阈值，低于此值的模块将有更多头被剪枝
    max_prune_ratio - 最大剪枝比例，避免过度剪枝
    """
    pruned_model = copy.deepcopy(model)
    
    # 获取每个 VisionSdpaAttention 模块的稀疏性
    sparse_scores = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, VisionSdpaAttention):  # VisionSdpaAttention类型检查
            # 获取当前模块的 qkv 权重
            qkv_weights = module.qkv.weight.data
            sparsity = calculate_sparsity(qkv_weights)
            sparse_scores.append((name, sparsity))
    
    # 根据稀疏性排序，稀疏性高的模块排在前面
    sparse_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 为每个模块决定剪枝数量
    for name, sparsity in sparse_scores:
        # 获取模块
        module = pruned_model.get_submodule(name)
        
        # 获取 qkv 和 proj 权重
        qkv_weights = module.qkv.weight.data
        proj_weights = module.proj.weight.data
        
        # 每个模块最多剪去的头数（最大为 4 个）
        num_heads = module.qkv.weight.size(0) // 3  # 假设 qkv 是 [3 * num_heads, embed_dim]
        max_heads_to_prune = num_heads // 4  # 每个模块最多剪掉 4 个头
        
        # 计算剪枝的头数：如果稀疏性大于阈值，剪掉更多头
        heads_to_prune_count = int(min(max_prune_ratio * num_heads, max_heads_to_prune))
        
        # 打印每个模块的稀疏性和剪枝头数
        print(f"Module {name} with sparsity {sparsity:.4f} will prune {heads_to_prune_count} heads.")
        
        # 选择剪枝的头：根据模块的稀疏性，优先剪掉最不重要的头
        for head_index in range(heads_to_prune_count):
            start_idx = head_index * proj_weights.shape[1] // num_heads
            end_idx = (head_index + 1) * proj_weights.shape[1] // num_heads
            qkv_weights[head_index] = 0  # 将 qkv 中的该头部分置零
            proj_weights[:, start_idx:end_idx] = 0  # 将 proj 中的该头权重置零
        
        # 重新赋值到模块中
        module.qkv.weight.data = qkv_weights
        module.proj.weight.data = proj_weights
    
    return pruned_model

def safe_prune_and_save(original_model, sparsity_threshold=0.5, max_prune_ratio=0.5, save_path="./saver/pruned_head-1"):
    """
    安全剪枝并保存模型的函数（不修改原始模型），基于稀疏性
    """
    # 基于稀疏性剪枝
    pruned_model = prune_heads_by_sparsity(original_model, sparsity_threshold, max_prune_ratio)
    
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 保存剪枝后的模型
    pruned_model.save_pretrained(save_path)
    
    # 还需要保存 AutoProcessor 配置
    processor = AutoProcessor.from_pretrained(original_model.config._name_or_path)
    processor.save_pretrained(save_path)
    
    print(f"Pruned model and processor saved to {save_path}")

# 使用示例
if __name__ == "__main__":
    # 加载原始模型
    original_model = Qwen2VLForConditionalGeneration.from_pretrained("./Qwen-2B-1")
    
    # 执行剪枝并保存
    safe_prune_and_save(
        original_model,
        sparsity_threshold=0.5,  # 设置稀疏性阈值
        max_prune_ratio=0.5,     # 设置最大剪枝比例
        save_path="./saver/pruned_head-1"  # 自动创建目录
    )
