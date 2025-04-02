from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import math

# R input N*d
def normalize(R):
    with torch.no_grad():
        mean = R.mean(dim=0)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True)
        R = R/norms
    return R

def cal_cov(R):
    with torch.no_grad():
        Z = torch.nn.functional.normalize(R, dim=1)
        A = torch.matmul(Z.T, Z)/Z.shape[0]
    return A

def cal_erank(A):
    with torch.no_grad():
        eig_val = torch.svd(A / torch.trace(A))[1] 
        entropy = - (eig_val * torch.log(eig_val)).nansum().item()
        erank = math.exp(entropy)
    return erank

def compute(R):
    return cal_erank(cal_cov(normalize(R)))

model_path_pre = "./Qwen-2B" # for example
tokenizer = AutoTokenizer.from_pretrained(model_path_pre)
model_1 = AutoModel.from_pretrained(model_path_pre).cuda()
config = AutoConfig.from_pretrained(model_path_pre)

model_path_post = "./saves/qwen2_vl-2b-bothml/full/sft" # for example
tokenizer_2 = AutoTokenizer.from_pretrained(model_path_post)
model_2 = AutoModel.from_pretrained(model_path_post).cuda()
config_2 = AutoConfig.from_pretrained(model_path_post)

# 输入文本
text = "We introduce a rank-based metric called Diff-eRank, which is rooted in information theory and geometry principles. Diff-eRank evaluates LLMs by examining their hidden representations to quantify how LLMs discard redundant information after training." # for example

# 将文本转换为模型输入张量
inputs = tokenizer(text, return_tensors="pt").to('cuda')

# 计算微调前（pre）和微调后（post）模型的隐藏表示
with torch.no_grad():
    # 微调前模型的隐藏表示
    R_pre = model_1(inputs.input_ids)[0][0, :, :]  # 取第一个样本的所有隐藏状态
    # 微调后模型的隐藏表示
    R_post = model_2(inputs.input_ids)[0][0, :, :]  # 取第一个样本的所有隐藏状态

# 计算微调前和微调后模型的有效秩
erank_pre = compute(R_pre)
erank_post = compute(R_post)

# 计算有效秩的变化
erank_diff = erank_post - erank_pre

# 输出结果
print(f"微调前模型的有效秩 (eRank_pre): {erank_pre}")
print(f"微调后模型的有效秩 (eRank_post): {erank_post}")
print(f"有效秩的变化 (eRank_diff): {erank_diff}")