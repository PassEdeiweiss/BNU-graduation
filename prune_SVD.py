import torch
import torch.nn as nn
import os
from copy import deepcopy
from transformers import AutoModel, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2MLP, Qwen2VLModel, VisionMlp, Qwen2VLForConditionalGeneration
)
# torch.set_printoptions(profile="full")

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


def prune_MLP_model(model: nn.Module, vision_prune_ratio: float = 0.45) -> tuple[nn.Module, dict]:
    pruned_intermediate_sizes = {}

    def prune_module(module, module_name, vision_prune_ratio = 0.45):
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

            # 实例化剪枝后的 MLP
            pruned_mlp = VisionMlp(
                dim=module.fc1.in_features,  # 输入维度保持不变
                hidden_dim=pruned_intermediate_size_fc2,  # 剪枝后的中间层维度
                hidden_act=hidden_act  # 继承原始激活函数
            )

            pruned_mlp.fc1.weight.data = module.fc1.weight[prune_mask, :].clone()
            pruned_mlp.fc2.weight.data = module.fc2.weight[:, prune_mask].clone()

            # pruned_mlp.fc2.weight.data = module.fc2.weight[prune_mask, :]
            # pruned_mlp.fc2.bias.data = module.fc2.bias[prune_mask]
            # pruned_mlp.fc1.weight.data = module.fc1.weight[: ,prune_mask]
            # pruned_mlp.fc1.bias.data = module.fc1.bias[prune_mask]

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

if __name__ == "__main__":
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained("./Qwen-2B-1")
    processor = AutoProcessor.from_pretrained("./Qwen-2B-1")
    
    # print_model_parameters(model)
    
    # 执行剪枝
    vision_prune_ratio = 0.75  # 剪枝比例
    pruned_model, pruned_sizes = prune_MLP_model(model, vision_prune_ratio=vision_prune_ratio)
    
    # 更新全局配置（假设所有MLP层剪枝比例相同）
    # if pruned_sizes:
    #     first_size = next(iter(pruned_sizes.values()))
    #     pruned_model.config.intermediate_size = first_size

    pruned_model.config.vision_config.mlp_ratio = 2.2
    # 保存模型
    save_dir = "saver/pruned_vision_model-SVD-2.2"
    os.makedirs(save_dir, exist_ok=True)
    
    # 更新并保存config
    pruned_model.config.to_json_file(os.path.join(save_dir, "config.json"))

    print(pruned_model)
    
    # 保存剪枝后的模型
    pruned_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir, use_fast=False)
    print("保存成功。")