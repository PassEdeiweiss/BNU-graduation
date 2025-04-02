from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import math

# 定义有效秩计算函数
def normalize(R):
    with torch.no_grad():
        R = R.float()  # 将张量转换为 Float32
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R / norms
    return R

def cal_cov(R):
    with torch.no_grad():
        R = R.float()  # 将张量转换为 Float32
        Z = torch.nn.functional.normalize(R, dim=1)
        A = torch.matmul(Z.mT, Z) / Z.shape[0]
    return A

def cal_erank(A):
    with torch.no_grad():
        A = A.float()  # 将张量转换为 Float32
        eig_val = torch.svd(A / torch.trace(A))[1]
        entropy = -(eig_val * torch.log(eig_val)).nansum().item()
        erank = math.exp(entropy)
    return erank

def compute(R):
    return cal_erank(cal_cov(normalize(R)))

# 加载微调前和微调后的模型及处理器
model_path_pre = "./Qwen-2B-1"  # 微调前模型路径
model_path_post = "./saver/pruned_vision_model-V"  # 微调后模型路径

# 加载模型和处理器
model_pre = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path_pre, torch_dtype=torch.float32, device_map="auto", local_files_only=True
)
model_post = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path_post, torch_dtype=torch.float32, device_map="auto", local_files_only=True
)

processor_pre = AutoProcessor.from_pretrained(model_path_pre)
processor_post = AutoProcessor.from_pretrained(model_path_post)

# 加载本地图像
image_path = "./test.jpg"  # 替换为本地图像路径
image = Image.open(image_path)

# 输入文本
text = "请阐释这张图片的内容。"

# 构建对话
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ],
    }
]

# 预处理输入
text_prompt_pre = processor_pre.apply_chat_template(conversation, add_generation_prompt=True)
text_prompt_post = processor_post.apply_chat_template(conversation, add_generation_prompt=True)

# 微调前模型的处理
inputs_pre = processor_pre(
    text=[text_prompt_pre], images=[image], padding=True, return_tensors="pt"
)
inputs_pre = inputs_pre.to("cuda")
inputs_pre["input_ids"] = inputs_pre["input_ids"].long()

# 微调后模型的处理
inputs_post = processor_post(
    text=[text_prompt_post], images=[image], padding=True, return_tensors="pt"
)
inputs_post = inputs_post.to("cuda")
inputs_post["input_ids"] = inputs_post["input_ids"].long()

# 提取文本隐藏表示
def extract_text_hidden_states(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        R_text = outputs.hidden_states[-1][0, :, :]
        print(R_text)
    return R_text

R_pre_text = extract_text_hidden_states(model_pre, inputs_pre)

R_post_text = extract_text_hidden_states(model_post, inputs_post)

erank_pre_text = compute(R_pre_text)
erank_post_text = compute(R_post_text)

erank_diff_text = erank_post_text - erank_pre_text

# 输出结果
print("全部模块的 eRank 变化:")
print(f"微调前: {erank_pre_text}")
print(f"微调后: {erank_post_text}")
print(f"变化值: {erank_diff_text}")