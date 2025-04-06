#!/bin/bash

# 模型路径列表
MODEL_PATHS=(
  "./Qwen-2B-1"
  "./saver/pruned_vision_model-SVD-3.5"
  "./saver/pruned_vision_model-SVD-3"
  "./saver/pruned_vision_model-SVD-2.8"
  "./saver/pruned_vision_model-SVD-2.6"
  "./saver/pruned_vision_model-SVD-2.4"
  "./saver/pruned_vision_model-SVD-2.2"
  "./saver/pruned_vision_model-SVD-2"
)

# 循环每个模型路径并运行 Python 脚本
for MODEL_PATH in "${MODEL_PATHS[@]}"
do
    echo "Running model with path: $MODEL_PATH"
    python test_dataset_MME.py --model_path "$MODEL_PATH"
done
