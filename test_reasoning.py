from PIL import Image
import torch
from transformers import AutoProcessor, AutoProcessor
from pruneclass import Qwen2VLPruned
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2MLP, Qwen2VLModel
)
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset

# class Qwen2VLPrunedForConditionalGeneration(Qwen2VLForConditionalGeneration):

# model_path = "./saver/pruned_vision_model-V"
# model_path = "./saver/pruned_vision_model-SVD"
# model_path = "./Qwen-2B-1"
# model_path = "./saver/sft-Qwen2VLpruned_merged"
# model_path = "./saver/qwen2_svd_train_full"
model_path = "./saver/pruned_vision_model-SVD-3.5"
# model_path = "./saver/pruned_vision_model-SVD-3"
# model_path = "./saver/pruned_vision_model-SVD-2.8"

dataset = load_dataset("AI4Math/MathVerse", "testmini")
dataset_text_only = load_dataset("AI4Math/MathVerse", "testmini_text_only")

model = Qwen2VLForConditionalGeneration.from_pretrained(model_path).to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

# print(dataset["testmini"][0])

for i in range(0, 10):
    image = dataset["testmini"][i]['image']
    text = dataset["testmini"][i]['query_cot']
    # print(text)

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

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}  # 确保所有数据都在 GPU

    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(output_text, dataset["testmini"][i]['answer'])