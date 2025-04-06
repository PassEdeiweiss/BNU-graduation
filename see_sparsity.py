import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, VisionMlp
from PIL import Image

# 加载模型和处理器
model_path = "./Qwen-2B-1"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path).to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

# 加载输入图片
image_path = "./test.jpg"
image = Image.open(image_path)

# 定义对话框
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "描述这张图片。"},
        ],
    }
]

# 创建输入的文本提示
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# 处理输入数据
inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}  # 确保数据都在 GPU

# 生成输出
output_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

def check_sparsity(model, threshold=1e-3):
    """
    通过模块类型判断视觉MLP层
    Args:
        model: 待检测模型
        threshold (float): 零值判定阈值
    Returns:
        dict: 视觉MLP层的稀疏度映射
    """
    layer_sparsities = {}
    
    # 遍历所有子模块
    for name, module in model.named_modules():
        # 类型匹配检查
        if isinstance(module, VisionMlp):  # 假设VisionMlp是视觉MLP层的类型
            # 获取该模块的所有线性层权重
            for weight_name, param in module.named_parameters():
                if 'weight' in weight_name:
                    full_name = f"{name}.{weight_name}"
                    
                    # 计算稀疏度
                    with torch.no_grad():
                        zero_mask = (torch.abs(param) < threshold)
                        non_zero = torch.sum(~zero_mask).item()
                        total = param.numel()
                        sparsity = 1.0 - non_zero / total
                    
                    layer_sparsities[full_name] = round(sparsity, 4)
    
    return layer_sparsities

# 获取稀疏性并输出
sparsity_info = check_sparsity(model)

# 可视化稀疏性
layer_names = list(sparsity_info.keys())
sparsities = list(sparsity_info.values())

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.barh(layer_names, sparsities, color='skyblue')
plt.xlabel('Sparsity')
plt.ylabel('Layer')
plt.title('Sparsity Visualization of Model Layers')

plt.savefig('./sparsity_mlp.png')


# 输出生成的文本
print("Generated Text: ", output_text)
