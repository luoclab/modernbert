import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base")
print(model.config.num_hidden_layers)  # 输出 Transformer 层数

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
print("token input",inputs)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(predicted_class_id)

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=9)
logits = model(**inputs).logits
output = model(**inputs,output_hidden_states=True)
hidden_states = output.hidden_states  # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
print("hidden_states.shape",len(hidden_states))
cls_embeddings = hidden_states[2][:, :, :]  # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
print("cls_embeddings.shape",cls_embeddings.shape)
print(logits)
for i in range(2, 13):
    cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
print(cls_embeddings.shape)

# predicted_class_id = logits.argmax().item()

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
print(loss)

