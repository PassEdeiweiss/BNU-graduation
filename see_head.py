import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionSdpaAttention, Qwen2VLForConditionalGeneration

def visualize_attention_head_sparsity(model, save_path="./x.jpg"):
    """
    可视化模型中注意力头权重的稀疏性
    参数:
    model - 需要进行可视化的模型
    """
    sparsity_scores = []  # 用于存储每个头的稀疏性得分
    
    # 遍历模型中的所有VisionSdpaAttention模块
    for name, module in model.named_modules():
        if isinstance(module, VisionSdpaAttention):  # 确保是VisionSdpaAttention类型的模块
            # 提取当前模块的 qkv 和 proj 权重
            qkv_weights = module.qkv.weight.data  # 形状 [3 * num_heads, embed_dim]
            proj_weights = module.proj.weight.data  # 形状 [embed_dim, embed_dim]

            # 计算每个注意力头的稀疏性，零元素比例
            num_heads = qkv_weights.shape[0] // 3  # 因为qkv有三个部分，每个部分对应头的数量
            head_dim = proj_weights.shape[1]  # 每个头的维度（embed_dim）

            for head_idx in range(num_heads):
                # 获取当前头的 qkv 权重（q、k、v 三部分是共享的，所以直接提取其中一部分即可）
                head_qkv_weights = qkv_weights[head_idx].reshape(-1)
                
                # 计算稀疏性：零元素比例
                num_zero_elements = (head_qkv_weights.abs() <= 0.01).sum().item()
                sparsity = num_zero_elements / head_qkv_weights.numel()
                sparsity_scores.append(sparsity)
                # print(sparsity_scores)
    
    # 将稀疏性得分转换为图形可视化格式
    # 生成一个热力图展示每个注意力头的稀疏性
    plt.figure(figsize=(12, 8))
    sns.heatmap([sparsity_scores], cmap="Blues", annot=True, fmt=".2f", cbar=True)
    plt.xlabel("Attention Heads")
    plt.ylabel("Sparsity")
    plt.title("Sparsity of Attention Head Weights")

    # 保存图像到指定路径
    plt.savefig(save_path)
    print(f"图像已保存至 {save_path}")
    plt.close()

# 示例：可视化已经加载的模型
if __name__ == "__main__":
    # 假设我们有一个已经加载的模型
    model = Qwen2VLForConditionalGeneration.from_pretrained("./Qwen-2B-1")
    
    # 可视化注意力头的稀疏性
    visualize_attention_head_sparsity(model)
