import torch
import torch.nn as nn
import torch.utils.data as Data
import config
from dataloader import get_data_v1_0
from model import ModernBert_TextCNN, TextCNN
import torch.optim as optim
from dataloader import MyDataset
import datetime
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
# import model_cls
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


print(torch.cuda.is_available())
batch_size = config.batch_size#64
epochs = config.epochs
idx2label = config.idx2label
learning_rate = config.learning_rate
weight_decay = config.weight_decay
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_curve = []





def caculate_recall(trues, preds):
    # preds = [0, 1, 2, 1, 0, 2, 0, 2, 0, 0, 2, 1, 2, 0, 1]
    # trues = [0, 1, 2, 1, 1, 2, 0, 2, 0, 0, 2, 1, 2, 1, 2]

    # 准确率 normalize=False则返回做对的个数
    acc = accuracy_score(trues, preds)
    acc_nums = accuracy_score(trues, preds, normalize=False)
    print(acc, acc_nums)  # 0.8 12

    # labels为指定标签类别，average为指定计算模式
    # micro-precision
    micro_p = precision_score(trues, preds, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='micro')
    # micro-recall
    micro_r = recall_score(trues, preds, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='micro')
    # micro f1-score
    micro_f1 = f1_score(trues, preds, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='micro')
    print(micro_p, micro_r, micro_f1)  # 0.8 0.8 0.8000000000000002

    # macro-precision
    macro_p = precision_score(trues, preds, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro')
    # macro-recall
    macro_r = recall_score(trues, preds, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro')
    # macro f1-score
    macro_f1 = f1_score(trues, preds, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro')
    print(macro_p, macro_r, macro_f1)
    return micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1,acc


def test(sentences_test, labels_test, model_path, result_file, filenames,DataParallel):
    import time
    s1 = time.time()
    file = open(result_file, 'w', encoding='utf-8')
 
    if DataParallel:
        loadnn = nn.DataParallel(ModernBert_TextCNN())
    else:
        loadnn = ModernBert_TextCNN()
    # loadnn = nn.DataParallel(Bert_Blend_CNN())

    state_dict = torch.load(model_path, map_location=device)
    loadnn.load_state_dict(state_dict=state_dict)
    loadnn.to(device)
    loadnn.eval()
    
    
    with torch.no_grad():
        test = MyDataset(sentences_test, labels=None, with_labels=False)
        print(f"test dataset has {len(test)} samples")
        label_pres ,scores = [], []
        pbar = tqdm(total=len(test))
        for i, test_text in enumerate(sentences_test):
            x = test.__getitem__(i)
            x = tuple(p.unsqueeze(0).to(device) for p in x)
            preds = loadnn([x[0], x[1], x[2]])
            # print(preds)
            pred = preds.data.max(dim=1, keepdim=True)[1]
            label_pre = int(pred[0][0])
            label_test = labels_test[i]
            filename = filenames[i]
            # print(label_pre, label_test, test_text)
            score__ = str(float(preds.data.max(dim=1, keepdim=True)[0]))
            test_text = test_text.replace('\n', '')          

            pre_text = idx2label[label_pre]
            label_text = idx2label[label_test]
            # print(pre_text, label_text)
            label_pres.append(label_pre)
            test_text = test_text[0:100]
            
            if len(test_text)<100:
                pre_text = '字数少于100'
            pbar.update(1)
            
        
        pbar.close()
        micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1,acc = caculate_recall(labels_test, label_pres)
        cm = confusion_matrix(labels_test, label_pres)
            # 绘制并保存混淆矩阵图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=True, yticklabels=True)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig("confusion_matrix1.png")
        file.write('\n' +
            'micro_p' + '\t' + 'micro_r' + '\t' + 'micro_f1' + '\t' + 'macro_p' + '\t' + 'acro_r' + '\t' + 'macro_f1' + '\t' + 'acc' + '\n')
        file.write(str(micro_p) + '\t' + str(micro_r) + '\t' + str(micro_f1) + '\t' + str(macro_p) + '\t' + str(
            macro_r) + '\t' + str(macro_f1) + '\t' + str(acc) + '\n')
    s2 = time.time()
    file.write('cost time: ' + str(s2 - s1))


if __name__ == '__main__':
    model_path = 'checkpoint/model_v2.0_np_250212_128_40_le5_10_512.pth'
    result_file = 'result/mbert_result_v2.0_np_250213_128_40_le5_512_2.txt'
    # os.ma
    testpath = '/data/train/devide_V2.0_0.1_1/test'
    DataParallel = False
    sentences_test, labels_test, filenames = get_data_v1_0(testpath)
    test(sentences_test, labels_test, model_path, result_file, filenames,DataParallel)


