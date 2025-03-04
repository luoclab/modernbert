import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import sys 
sys.path.append("/add_cls_project/modernbert")
import config
import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification



num_filters = config.num_filters
bert_model = config.bert_model
hidden_size = config.hidden_size
n_class = config.n_class

encode_layer = config.encode_layer
filter_sizes = config.filter_sizes

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filter_total = num_filters * len(filter_sizes)#9
        self.Weight = nn.Linear(self.num_filter_total, n_class, bias=False)
        self.bias = nn.Parameter(torch.ones([n_class]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, num_filters, kernel_size=(size, hidden_size)) for size in filter_sizes
        ])

    def forward(self, x):
        # x: [bs, seq, hidden]
        x = x.unsqueeze(1)  # [bs, channel=1, seq, hidden]

        pooled_outputs = []
        # print("x.shape",x.shape)
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))  # [bs, channel=1, seq-kernel_size+1, 1]
            # print(f"h con {i}")
            # print("h.shape",h.shape)
            mp = nn.MaxPool2d(
                kernel_size=(encode_layer - filter_sizes[i] + 1, 1)
            )
            # mp: [bs, channel=3, w, h]
            pooled = mp(h).permute(0, 3, 2, 1)  # [bs, h=1, w=1, channel=3]
            # print("pooled.shape",pooled.shape)
            pooled_outputs.append(pooled)

        
        h_pool = torch.cat(pooled_outputs, len(filter_sizes))  # [bs, h=1, w=1, channel=3 * 3]
        # print("h_pool",h_pool.shape)
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
        # print("h_pool_flat.shape",h_pool_flat.shape)

        output = self.Weight(h_pool_flat) + self.bias  # [bs, n_class]

        return output

 
# model
class ModernBert_TextCNN(nn.Module):
    def __init__(self):
        super(ModernBert_TextCNN, self).__init__()
        self.modernbert = ModernBertForSequenceClassification.from_pretrained("modernbert_base",  output_hidden_states=True, return_dict=True,reference_compile=False)#,num_labels=config.n_class
        self.linear = nn.Linear(hidden_size, n_class)
        self.textcnn = TextCNN()

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.modernbert(input_ids=input_ids, attention_mask=attention_mask,
                           )  # 返回一个output字典
        # outputs = self.modernbert(**X
        #                    )
        # 取每一层encode出来的向量
        # outputs.pooler_output: [bs, hidden_size]
        hidden_states = outputs.hidden_states  # 23*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
        for i in range(2, 23):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        # print("cls_embeddings.shape",cls_embeddings.shape)
        logits = self.textcnn(cls_embeddings)
        # print(" logits", logits.shape)
        return logits

if __name__=="__main__":
    model_cls=ModernBert_TextCNN()
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # print(inputs)
    logits=model_cls(inputs)
    print(logits.shape)
