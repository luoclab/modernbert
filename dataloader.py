# -*- coding: utf-8 -*-            
# @Author : gongyutian@wps.cn
# @Time : 2023/10/31 上午8:52
import os
from random import shuffle
import torch.utils.data as Data
from transformers import AutoModel, AutoTokenizer
import config
import data_process

bert_model = config.bert_model
label2idx = config.label2idx
# laber2idx = {'简历': 0, '试卷': 1}
maxlen = config.maxlen

class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True, ):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = self.sentences[index]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects

        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

def get_data_old(path):
    texts,labels,filename = [] ,[], []
    filelist = os.listdir(path)
    for file in filelist:
        filename.append(file)
        labels_ = label2idx.keys()
        for lab in labels_:
            if lab in file:
                label = label2idx[lab]
                break
        filepath = os.path.join(path,file)
        filereader = open(filepath,'r',encoding='utf-8')
        text = ''
        for line in filereader:
            text += line.strip('\n')
        text = data_process.data_pro(text)
        if len(text) < 50:
            print(file)
            continue
        texts.append(text)
        labels.append(label)
    c = list(zip(texts, labels,filename))
    shuffle(c)
    texts, labels, filename = zip(*c)
    return texts,labels,filename


def get_data_v1_0(path):
    texts, labels,filenames = [], [], []
    for label,idx in label2idx.items():
        dir_path = os.path.join(path,label)
        if not os.path.exists(dir_path):
            continue
        for file in os.listdir(dir_path):
            filepath = os.path.join(dir_path, file)
            filename_ = file.split('.txt')[0]
            if os.path.isdir(filepath):
                continue
            filereader = open(filepath, 'r', encoding='utf-8')
            text = ''
            for line in filereader:#逐行读取，去掉换行符
                text += line.strip('\n')
            text = data_process.data_pro(text)
            #if len(text) < 50:
                #print(file)
                #continue
            texts.append(text)
            labels.append(label2idx[label])
            filenames.append(filename_)
    c = list(zip(texts, labels,filenames))
    #c = [(text1, label1, filename1), (text2, label2, filename2), ...]
    shuffle(c)
    texts, labels,filenames = zip(*c)#包回 texts、labels、filenames，但它们变成了 元组
    texts, labels,filenames = list(texts), list(labels), list(filenames)#转换回列表
    return texts, labels, filenames


if __name__ == '__main__':
    trainpath = 'data/train/train_V1.0_all'
    # print(trainpath, model_path)
    sentences_train, labels_train, _ = get_data_v1_0(trainpath)
    dataset=MyDataset(sentences_train, labels_train)
    print(f" dataset has {len(dataset)} samples")
    print("done")
    # train_ = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=True,
    #                          num_workers=1)
    # for i in train_:
    #     batch = tuple(p for p in batch)
    # a = [1, 2, 3, 4]
    # b = [11, 22, 33, 44]
    # c = list(zip(a, b))
    # shuffle(c)
    # a, b = zip(*c)
    # print(a,b)