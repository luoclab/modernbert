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



# model
class ModernBert(nn.Module):
    def __init__(self):
        super(ModernBert, self).__init__()
        self.modernbert = ModernBertForSequenceClassification.from_pretrained("modernbert_base",  output_hidden_states=True, return_dict=True,reference_compile=False,num_labels=config.n_class)#,num_labels=config.n_class


    def forward(self, X):
        # input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.modernbert(**X                                 
                           )  # 返回一个output字典

        return outputs.logits

if __name__=="__main__":
    model_cls=ModernBert()
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # print(inputs)
    logits=model_cls(inputs)
    print(logits.shape)
