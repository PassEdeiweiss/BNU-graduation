import pyarrow.parquet as pq
import pandas as pd
import torch
from transformers import AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from PIL import Image
import os
import argparse
import gc

# 解析命令行参数
parser = argparse.ArgumentParser(description="Test Qwen2VL model with images and text")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
parser.add_argument("--start_idx", type=int, default=0, help="起始样本索引")
parser.add_argument("--end_idx", type=int, default=25, help="结束样本索引")
args = parser.parse_args()

# 使用命令行参数中的 model_path
model_path = args.model_path

# 确保内存清理
torch.cuda.empty_cache()
gc.collect()

# 设置低精度推理以减少内存使用
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用低精度和优化设置加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map="auto",  # 自动管理设备映射
    low_cpu_mem_usage=True,  # 减少CPU内存使用
)
processor = AutoProcessor.from_pretrained(model_path)

# 从parquet文件读取数据
parquet_file = pq.ParquetFile('./MME/data/test-00000-of-00002.parquet')
data = parquet_file.read().to_pandas()

# 只处理指定范围的样本
sampled_data = data.iloc[args.start_idx:args.end_idx]

correct_predictions = 0
total_predictions = 0

# 为所有进程设置
with torch.inference_mode():  # 使用inference_mode代替no_grad以进一步优化
    for i, row in sampled_data.iterrows():
        # 在每个循环开始时清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        image_path = "./MME/down/" + row['question_id'] + ".jpg"
        
        # 检查图片文件的扩展名是否是 png 或 jpg
        if not os.path.exists(image_path):
            image_path = "./MME/down/" + row['question_id'] + ".png"
        
        try:
            image = Image.open(image_path)
            image = image.convert("RGB")  # 转换为 RGB 格式，确保输入一致
            text = row['question']

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            ]

            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # 创建输入但直接放到设备上
            inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 使用更低的max_new_tokens减少内存使用
            # 使用generation_config参数优化生成过程
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=64,  # 减小此值以减少内存使用
                do_sample=False,  # 使用贪婪解码而不是采样
                pad_token_id=processor.tokenizer.pad_token_id,
                num_beams=1  # 使用贪婪搜索而不是beam search
            )
            
            # 提取生成的部分
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # 处理模型输出和真实答案，去掉尾部的句点
            model_output = output_text[0].strip().lower().rstrip('.')  # 去掉尾部的句点
            correct_answer = row['answer'].strip().lower().rstrip('.')  # 去掉尾部的句点

            print(f"预测: {model_output}, 正确答案: {correct_answer}")

            if model_output == correct_answer:
                correct_predictions += 1
            total_predictions += 1
            print(f"已处理: {total_predictions}，当前正确率: {correct_predictions/total_predictions:.4f}")

            # 清理内存
            del inputs
            del output_ids
            del generated_ids
            
        except Exception as e:
            print(f"处理样本时出错: {e}")
            continue

# 计算准确率
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

# 保存结果
output_file = './FLOPs/Qwen-MME.txt'
with open(output_file, 'a') as f:
    f.write(f"{model_path}\n")
    f.write(f"总样本数: {total_predictions}, 正确预测: {correct_predictions}\n")
    f.write(f"准确率: {accuracy:.4f}\n")
    f.write("-" * 50 + "\n")

# 最终清理
del model
del processor
torch.cuda.empty_cache()
gc.collect()