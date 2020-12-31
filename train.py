import os
import torch
import torch.nn as nn
import torch.optim as optim
from  torch.utils.data import DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from dataset import dataset
from model import BiLSTM_CRF
from config import *

torch.manual_seed(1)  # 最后的路径正确，但是分值无法稳定复现，原因不明
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tag2ix = {"B-PER": 0, "I-PER": 1,
                  "B-LOC": 2, "I-LOC": 3,
                  "B-ORG": 4, "I-ORG": 5, "O": 6, 'START': 7, 'STOP': 8}
ix2tag = {v:k for k,v in tag2ix.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == "__main__":
    train_dataset = dataset(mode='train',seq_length=args.max_seq_length)
    test_dataset = dataset(mode='test',seq_length=args.max_seq_length)
    train_data_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=False)
    test_data_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,drop_last=False)


    model = BiLSTM_CRF(tag2ix,train_dataset.voc2idx,device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(args.epochs):
        print('epoch:%s' % epoch)
        pbar = tqdm(train_data_loader)
        for data in pbar:
            words,tags,sent_length = data
            words = words.to(device)
            tags = tags.to(device)
            sent_length = sent_length.to(device)
            #print(words.shape)
            model.zero_grad()  # PyTorch默认会累积梯度; 而我们需要每条样本单独算梯度
            #loss,paths = model(words,tags,train_flag=True,decode_flag=False)
            loss = model.neg_log_likelihood(words,tags,sent_length)
            loss.backward()  # 前向求出负对数似然(loss); 然后回传梯度
            optimizer.step()  # 梯度下降，更新参数
            pbar.set_description("Loss: %s" % loss.item())
        state = {
            'state': model.state_dict(),
            'epoch': epoch  # 将epoch一并保存
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/bilstm_crf.t7')

    # 训练后的预测结果(有意义的结果，与label一致); 打印类似 (18.722553253173828, [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
        if (epoch+1) % args.interval_save == 0:
            with torch.no_grad():  # 这里用了第一条训练数据(而非专门的测试数据)，仅作教学演示
                pbar = tqdm(test_data_loader)
                all_pre = []
                all_tru_tag = []
                for data in pbar:
                    words, tags, sent_length = data
                    words = words.to(device)
                    #valid_loss,predict_path = model(words,tags,train_flag=False,decode_flag=True)
                    predict_path = model(words,sent_length)
                    tags = tags.numpy().tolist()

                    batch_pre_path = []
                    batch_tru_tag = []
                    #bt,s_l = predict_path.size()
                    bt = len(predict_path)
                    s_l = len(predict_path[0])
                    for i in range(bt):
                        path = predict_path[i]
                        tru_path = tags[i]
                        predict_tags = []
                        tru_tags = []
                        for j in range(s_l):
                            pre_tag = path[j].item()
                            tru_tag = tru_path[j]
                            predict_tags.append(ix2tag[pre_tag])
                            tru_tags.append(ix2tag[tru_tag])
                        predict_tags = predict_tags[:sent_length[i]]
                        tru_tags = tru_tags[:sent_length[i]]
                        batch_pre_path.append(predict_tags)
                        batch_tru_tag.append(tru_tags)

                    all_pre += batch_pre_path
                    all_tru_tag += batch_tru_tag
                    #pbar.set_description("loss: %s" % (valid_loss))


                print('save..')
                t = ''
                with open('test_predict_tags.txt', 'w', encoding='utf-8') as f:
                    for i in all_pre:
                        for e in range(len(i)):
                            t = t + str(i[e]) + ' '
                        f.write(t.strip(' '))
                        f.write('\n')
                        t = ''

                acc = accuracy_score(all_tru_tag,all_pre)
                prec = precision_score(all_tru_tag,all_pre)
                recall = recall_score(all_tru_tag,all_pre)
                f1 = f1_score(all_tru_tag,all_pre)
                print('acc:%.3f, prec:%.3f,recall:%.3f,f1:%.3f'%(acc,prec,recall,f1))
