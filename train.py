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
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

print(torch.cuda.is_available())
batch_size = config.batch_size#64
epochs = config.epochs
idx2label = config.idx2label
learning_rate = config.learning_rate
weight_decay = config.weight_decay
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_curve = []


def train(sentences_t, labels_t, model_path):
    min_sum = 1000000
    bert_blend_cnn = ModernBert_TextCNN()
    # bert_blend_cnn = nn.DataParallel(bert_blend_cnn) #最基础的bert
    bert_blend_cnn.to(device)
    optimizer = optim.Adam(bert_blend_cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_ = Data.DataLoader(dataset=MyDataset(sentences_t, labels_t), batch_size=batch_size, shuffle=True,
                             num_workers=1)
    # train
    sum_loss = 0
    total_step = len(train_)
    print(epochs, batch_size)
    # pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        for i, batch in enumerate(train_):
            optimizer.zero_grad()
            batch = tuple(p.to(device) for p in batch)
            pred = bert_blend_cnn([batch[0], batch[1], batch[2]])
            loss = loss_fn(pred, batch[3])
            sum_loss += loss.item()

            loss.backward()
            optimizer.step()
            # if epoch % 10 == 0:
            timenow = datetime.datetime.now()
            print(
                '{}:[{}|{}] step:{}/{} loss:{:.4f}'.format(timenow, epoch + 1, epochs, i + 1, total_step, loss.item()))
        print('epoch:{},loss:{}'.format(epoch + 1, sum_loss))
        train_curve.append(sum_loss)
        if epoch > 20:
            if sum_loss < min_sum:
                min_sum = sum_loss
                torch.save(bert_blend_cnn.state_dict(), model_path)
                # torch.save(bert_blend_cnn.state_dict(), model_path)
                print('save...', epoch + 1)
        # if epoch == 60:
        #         torch.save(bert_blend_cnn.state_dict(), model_path.split('.pth')[0]+'_60.pth')
        #         print('save...', epoch + 1)
        # if epoch == 90:
        #         torch.save(bert_blend_cnn.state_dict(), model_path.split('.pth')[0]+'_90.pth')
        #         print('save...', epoch + 1)
        train_curve.append(sum_loss)
        sum_loss = 0
    #     pbar.update(1)
            
        
    # pbar.close()


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
    return micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1


def test(sentences_test, labels_test, model_path, result_file, filenames,DataParallel):
    import time
    s1 = time.time()
    file = open(result_file, 'w', encoding='utf-8')
    # file.write('文档id' + '\t' +'原始标签' + '\t' + '预测标签' + '\t' + '原始类别' + '\t' + '预测类别' + '\t'+ '预测分数' + '\t' + '文本' + '\n')

    #     loadnn = torch.load(model_path)
    # loadnn = model_cls.Bert_Blend_CNN()
    # state_dict = torch.load(textcls_model,map_location=device)
    # loadnn.load_state_dict(state_dict=state_dict) 
    # loadnn.to(device)
    # loadnn.eval()
    if DataParallel:
        loadnn = nn.DataParallel(ModernBert_TextCNN)
    else:
        loadnn = ModernBert_TextCNN
    # loadnn = nn.DataParallel(Bert_Blend_CNN())

    state_dict = torch.load(model_path, map_location=device)
    loadnn.load_state_dict(state_dict=state_dict)
    loadnn.to(device)
    loadnn.eval()
    # path = './data/test'
    # sentences_test, labels_test, filename_test = get_data_v1_0(path)

    # test
    
    
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
#                 label_pre = 6   
            
            # file.write(str(filename) + '\t' + str(label_test) + '\t' + str(
            #     label_pre) + '\t' + label_text + '\t' + pre_text + '\t' + score__ + '\t' + test_text + '\n')
            pbar.update(1)
            
        
        pbar.close()
        micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1,acc = caculate_recall(labels_test, label_pres)
        file.write('\n' +
            'micro_p' + '\t' + 'micro_r' + '\t' + 'micro_f1' + '\t' + 'macro_p' + '\t' + 'acro_r' + '\t' + 'macro_f1' + '\n')
        file.write(str(micro_p) + '\t' + str(micro_r) + '\t' + str(micro_f1) + '\t' + str(macro_p) + '\t' + str(
            macro_r) + '\t' + str(macro_f1) + '\t' + str(acc) + '\n')
    s2 = time.time()
    file.write('cost time: ' + str(s2 - s1))


if __name__ == '__main__':
    model_path = 'checkpoint/model_v2.0_np_250212_128_40_le5_10_512.pth'
    train_flag = True
    if train_flag:
        trainpath = '/data/train/devide_V2.0_0.1_1/train'
        print(trainpath, model_path)
        sentences_train, labels_train, _ = get_data_v1_0(trainpath)
        train(sentences_train, labels_train, model_path)
        print(train_curve)   

