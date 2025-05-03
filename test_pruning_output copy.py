from PIL import Image
import torch
import os
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLConfig,
    AutoProcessor
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionMlp
from safetensors.torch import load_file

class Qwen2VLPrunedForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """支持视觉MLP剪枝的Qwen2VL模型"""
    
    def __init__(self, config):
        # 配置验证
        assert hasattr(config.vision_config, "mlp_size"), "vision_config必须包含mlp_size参数"
        assert len(config.vision_config.mlp_size) == config.vision_config.depth, \
            f"mlp_size长度({len(config.vision_config.mlp_size)})与层数({config.vision_config.depth})不匹配"
        
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, pruned_model_path, *args, **kwargs):
        """加载基础模型并应用剪枝配置"""
        
        # 1. 加载原始模型配置
        base_config = Qwen2VLConfig.from_pretrained(pretrained_model_name_or_path)
        
        # 2. 加载剪枝配置
        pruned_config = Qwen2VLConfig.from_pretrained(pruned_model_path)
        print("剪枝模型配置验证：\n", pruned_config.vision_config.mlp_size)
        
        # 3. 初始化基础模型
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            config=base_config,  # 使用原始配置初始化
            **kwargs
        )
        
        # 4. 替换视觉MLP层
        for layer_idx in range(pruned_config.vision_config.depth):
            model.visual.blocks[layer_idx].mlp = VisionMlp(
                dim=pruned_config.vision_config.embed_dim,
                hidden_dim=pruned_config.vision_config.mlp_size[layer_idx],
                hidden_act=pruned_config.vision_config.hidden_act
            ).to(dtype=torch.bfloat16)
        
        # 5. 加载剪枝权重
        model.load_state_dict(
            torch.load(os.path.join(pruned_model_path, "pytorch_model_pruned.bin")),  # 使用合并后的权重文件
            strict=False
        )
        
        return model

if __name__ == "__main__":

    base_model_path = "./Qwen-2B-1"
    # pruned_model_path = "./Qwen-2B-1"
    # pruned_model_path = "./saver/pruned_vision_model-Qwen-threshold-pruning-2"
    pruned_model_path = "./saver/pruned_vision_model-eff-svd"

    processor = AutoProcessor.from_pretrained(
        pruned_model_path,
        use_fast=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen2VLPrunedForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        pruned_model_path=pruned_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).to(device)
    model.eval()

    print("模型加载完成，结构验证：")
    print(model)
    
    image_path = "./test.jpg"
    image = Image.open(image_path)
    
    print(f"图像已加载，尺寸: {image.size}")
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "描述这张图片。"},
            ],
        }
    ]
    
    text_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )
    
    print("对话模板已应用")
    
    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(output_text)