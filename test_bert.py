import torch
from transformers import AutoModel, AutoTokenizer

# 加载 BERT 模型（不包含分类头）
bert = AutoModel.from_pretrained("bert_model/bert-base-chinese", output_hidden_states=True, return_dict=True)
tokenizer = AutoTokenizer.from_pretrained("bert_model/bert-base-chinese")

# 进行文本编码
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# 前向传播
output = bert(**inputs)

# 获取 hidden_states
hidden_states = output.hidden_states  # 13 层（第一层是 embedding 层）

# 打印 hidden_states 层数
print("Hidden states 层数:", len(hidden_states))  # 预期输出 13
