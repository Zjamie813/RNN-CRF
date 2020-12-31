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

torch.manual_seed(1)  # 最后的路径正确，但是分值无法稳定复现，原因不明
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
START_TAG, END_TAG = "<START>", "<END>"
tag2ix = {"B-PER": 0, "I-PER": 1,
                  "B-LOC": 2, "I-LOC": 3,
                  "B-ORG": 4, "I-ORG": 5, "O": 6, START_TAG: 7, END_TAG: 8}
ix2tag = {v:k for k,v in tag2ix.items()}
batch_size = 512
max_seq_length = 100
epochs = 100
interval_save = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_sum_exp(smat):
    vmax = smat.max(dim=1, keepdim=True)# 每一列的最大数
    vmax = vmax[0] # max()函数得到的是tensor组成的tuple  # [128,1,9]
    # dev = smat - vmax
    # ep = dev.exp()
    # su = ep.sum(dim=1,keepdim=True)
    # lo = su.log()
    # ad = (lo + vmax).squeeze(1)
    return ((smat - vmax).exp().sum(dim=1, keepdim=True).log() + vmax).squeeze(1)


class BiLSTM_CRF(nn.Module):
    def __init__(self, tag2ix, word2ix, embedding_dim, hidden_dim):
        """
        :param tag2ix: 序列标注问题的 标签 -> 下标 的映射
        :param word2ix: 输入单词 -> 下标 的映射
        :param embedding_dim: 喂进BiLSTM的词向量的维度,dim=5
        :param hidden_dim: 期望的BiLSTM输出层维度,dim=4
        """
        super(BiLSTM_CRF, self).__init__()
        assert hidden_dim % 2 == 0, 'hidden_dim must be even for Bi-Directional LSTM'
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.tag2ix, self.word2ix, self.n_tags = tag2ix, word2ix, len(tag2ix)

        self.word_embeds = nn.Embedding(len(word2ix), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,num_layers=2, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.n_tags)  # 用于将LSTM的输出 降维到 标签空间
        # tag间的转移score矩阵，即CRF层参数; 注意这里的定义是未转置过的，即"i到j"的分数(而非"i来自j")
        self.transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))
        # "START_TAG来自于?" 和 "?来自于END_TAG" 都是无意义的
        self.transitions.data[:, tag2ix[START_TAG]] = self.transitions.data[tag2ix[END_TAG], :] = -10000
        self.dropout = nn.Dropout(0.1)

    def neg_log_likelihood(self, words, tags):  # 求一对 <sentence, tags> 在当前参数下的负对数似然，作为loss
        frames = self._get_lstm_features(words)  # emission score at each frame
        new_tags = tags.permute(1,0)
        gold_score = self._score_sentence(frames, new_tags)  # 正确路径的分数
        forward_score = self._forward_alg(frames)  # 所有路径的分数和
        return (forward_score - gold_score).mean()
        # for bt_id in range(batch_size):
        #    frame = frames[bt_id] # 一句话生成的
        #    tag = tags[bt_id]
        #    gold_score = self._score_sentence(frame, tag)  # 正确路径的分数
        #    forward_score = self._forward_alg(frame)  # 所有路径的分数和
        #    # -(正确路径的分数 - 所有路径的分数和）;注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)
        #    losses[bt_id] = forward_score - gold_score
        # return losses.mean()

    def _get_lstm_features(self, words):  # 求出每一帧对应的隐向量
        # LSTM输入形状(seq_len, batch=1, input_size); 教学演示 batch size 为1
        embeds = self.word_embeds(words) # [bt,seq_length,dim]
        embeds = embeds.permute(1,0,2)
        # 随机初始化LSTM的隐状态H
        #hidden = torch.randn(2, batch_size, self.hidden_dim // 2), torch.randn(2, batch_size, self.hidden_dim // 2)
        lstm_out, _hidden = self.lstm(embeds)  # lstm_out:[batch,seq_len,hidden_dim],是hidden维度的输出,hidden:两个[2,1,2]
        return self.hidden2tag(lstm_out)  # 把LSTM输出的隐状态张量去掉batch维，然后降维到tag空间

    def _score_sentence(self, frames, tags):
        """
        求路径pair: frames->tags 的分值
        index:      0   1   2   3   4   5   6
        frames:     F0  F1  F2  F3  F4
        tags:  <s>  Y0  Y1  Y2  Y3  Y4  <e>
        """
        seq_length,bt,tag_size = frames.size()

        # end_tag = torch.full((1,bt),tag2ix[END_TAG],dtype=torch.int,requires_grad=False).to(device)
        start_tag = torch.full((1, bt), tag2ix[START_TAG],dtype=torch.int,requires_grad=False).to(device)
        # #tags_tensor = self._to_tensor([START_TAG] + tags, self.tag2ix)  # 注意不要+[END_TAG]; 结尾有处理
        # #tags_tensor = tags #[seq_len,batchsize]
        #
        tags_tensor = torch.cat([start_tag, tags], dim=0)
        #tags_tensor = torch.cat([start_tag,tags],dim=0).unsqueeze(-1).unsqueeze(-1)
        # seq_curr_tag = torch.cat([tags,end_tag],dim=0).unsqueeze(-1).unsqueeze(-1).expand(seq_length+1,bt,1,tag_size)
        # batch_trans_mat = self.transitions.unsqueeze(0).unsqueeze(0).expand(seq_length + 1, bt, tag_size, tag_size)
        #
        # trans_score = torch.gather(batch_trans_mat,dim=2,index=seq_curr_tag)
        # trans_score = torch.gather(trans_score,dim=3,index=seq_prev_tag) #这里还是在定位转移值
        # batch_trans_score = trans_score.sum(dim=0).squeeze(-1).squeeze(-1)
        #
        # stat_score = torch.gather(frames,dim=2,index=tags.unsqueeze(-1))
        # batch_stat_score = stat_score.sum(dim=0).squeeze(-1)
        # return batch_trans_score + batch_stat_score
        #把循环变成了gather
        score = torch.zeros(bt).to(device)
        for i, frame in enumerate(frames):  # 沿途累加每一帧的转移和发射 #[128,9]
            score += self.transitions[tags_tensor[i], tags_tensor[i + 1]] + frame[range(bt), tags_tensor[i + 1]]
        return score + self.transitions[tags_tensor[-1], self.tag2ix[END_TAG]]  # 加上到END_TAG的转移

    def _forward_alg(self, frames):
        """ 给定每一帧的发射分值; 按照当前的CRF层参数算出所有可能序列的分值和，用作概率归一化分母 """
        seq_length,bt,tag_size = frames.size()
        alpha = torch.full((bt, self.n_tags), -10000.0).to(device) #[128,9]
        alpha[:,self.tag2ix[START_TAG]] = 0  # 初始化分值分布. START_TAG是log(1)=0, 其他都是很小的值 "-10000"
        batch_trans_mat = self.transitions.unsqueeze(0).expand(bt, tag_size, tag_size)
        for frame in frames: #frame:[128,9]
            # log_sum_exp()内三者相加会广播: 当前各状态的分值分布(列向量) + 发射分值(行向量) + 转移矩阵(方形矩阵)
            # 相加所得矩阵的物理意义见log_sum_exp()函数的注释; 然后按列求log_sum_exp得到行向量
            # a = alpha.unsqueeze(2).repeat(1,1,self.n_tags) #[128,9,9]
            # b = frame.unsqueeze(1).repeat(1,self.n_tags,1) # [128,9,9]
            # c = self.transitions.unsqueeze(0).repeat(batch_size,1,1)
            #d = a + b + c # [128,9,9]
            # d = alpha.unsqueeze(-1).expand(bt,tag_size,tag_size)+\
            #                     frame.unsqueeze(1).expand(bt,tag_size,tag_size)+batch_trans_mat
            alpha = log_sum_exp(alpha.unsqueeze(-1).expand(bt,tag_size,tag_size)+\
                                frame.unsqueeze(1).expand(bt,tag_size,tag_size)+batch_trans_mat) # [128,9,9]

            #alpha = log_sum_exp(alpha.t() + frame + self.transitions)
        # 最后转到EOS，发射分值为0，转移分值为列向量 self.transitions[:, [self.tag2ix[END_TAG]]]
        final = alpha.unsqueeze(-1) + 0 + self.transitions[:, [self.tag2ix[END_TAG]]].unsqueeze(0).expand(bt,-1,-1)
        return  log_sum_exp(final).flatten()
        #return log_sum_exp(alpha.t() + 0 + self.transitions[:, [self.tag2ix[END_TAG]]]).flatten()

    def _viterbi_decode(self, frames):
        seq_len,bt,tag_size = frames.size()
        backtrace = []  # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        alpha = torch.full((bt, self.n_tags), -10000.).to(device)
        alpha[:,self.tag2ix[START_TAG]] = 0
        for frame in frames: # frame:[5],就是tag dim
            # 这里跟 _forward_alg()稍有不同: 需要求最优路径（而非一个总体分值）, 所以还要对smat求column_max
            # a = alpha.unsqueeze(2).repeat(1, 1, self.n_tags)  # [128,9,9]
            # b = frame.unsqueeze(1).repeat(1, self.n_tags, 1)  # [128,9,9]
            # c = self.transitions.unsqueeze(0).repeat(batch_size, 1, 1)
            smat = alpha.unsqueeze(2).expand(bt, tag_size, tag_size) + \
                   frame.unsqueeze(1).expand(bt, tag_size, tag_size) + \
                   self.transitions.unsqueeze(0).expand(bt, tag_size, tag_size) #[128,9,9]
            backtrace.append(smat.argmax(1)) #[128,9]
            # smat = alpha.t() + frame.unsqueeze(0) + self.transitions  # [5,5]
            # backtrace.append(smat.argmax(0))  # 当前帧每个状态的最优"来源"
            alpha = log_sum_exp(smat)  # 转移规律跟 _forward_alg()一样; 只不过转移之前拿smat求了一下回溯路径

        # 回溯路径
        # [128,9,1]
        final = alpha.unsqueeze(2) + 0 + self.transitions[:, [self.tag2ix[END_TAG]]].unsqueeze(0).expand(bt, -1,-1)
        best_tag_id = final.argmax(1) # [128,1]
        ts_backtrace = torch.tensor([bc.tolist() for bc in backtrace])
        bt_pre_paths = []
        for sid in range(bt):
            s_backtrace = ts_backtrace[:,sid,:].squeeze(1)
            current_best_tag_id = best_tag_id[sid,0].item()
            pre_path = [current_best_tag_id]
            for bptrs_t in reversed(s_backtrace[1:]):  # 从[1:]开始，去掉开头的 START_TAG
                current_best_tag_id = bptrs_t[current_best_tag_id].item()
                pre_path.append(current_best_tag_id)
            bt_pre_paths.append(pre_path)

        # smat = alpha.t() + 0 + self.transitions[:, [self.tag2ix[END_TAG]]]
        # best_tag_id = smat.flatten().argmax().item()
        # best_tag_id = best_tag_id % self.n_tags
        # best_path = [best_tag_id]
        # for bptrs_t in reversed(backtrace[1:]):  # 从[1:]开始，去掉开头的 START_TAG
        #     best_tag_id = bptrs_t[best_tag_id].item()
        #     best_path.append(best_tag_id)
        return log_sum_exp(final).mean().item(), bt_pre_paths  # 返回最优路径分值 和 最优路径

    def forward(self, words):  # 模型inference逻辑
        lstm_feats = self._get_lstm_features(words)  # 求出每一帧的发射矩阵 shape[11,5]

        return self._viterbi_decode(lstm_feats)  # 采用已经训好的CRF层, 做维特比解码, 得到最优路径及其分数


if __name__ == "__main__":
    # training_data = [("the wall street journal reported today that apple corporation made money".split(),
    #                   "B I I I O O O B I O O".split()),
    #                  ("georgia tech is a university in georgia".split(), "B I O O O O B".split())]
    train_dataset = dataset(mode='train',seq_length=max_seq_length)
    test_dataset = dataset(mode='test',seq_length=max_seq_length)
    train_data_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    test_data_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)

    from crf import CRFLayer
    # model = CRFLayer(word2ix=train_dataset.voc2idx,embedding_dim=100,
    #                  hidden_size=512,num_entity_labels=6).to(device)


    model = BiLSTM_CRF(tag2ix=tag2ix,word2ix= train_dataset.voc2idx,
                       embedding_dim=300, hidden_dim=512).to(device)


    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(epochs):
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
            loss = model.neg_log_likelihood(words,tags)
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
        if (epoch+1) % interval_save == 0:
            with torch.no_grad():  # 这里用了第一条训练数据(而非专门的测试数据)，仅作教学演示
                pbar = tqdm(test_data_loader)
                all_pre = []
                all_tru_tag = []
                for data in pbar:
                    words, tags, sent_length = data
                    words = words.to(device)
                    #valid_loss,predict_path = model(words,tags,train_flag=False,decode_flag=True)
                    valid_loss,predict_path = model(words)
                    tags = tags.numpy().tolist()

                    batch_pre_path = []
                    batch_tru_tag = []

                    bt = len(predict_path)
                    s_l = len(predict_path[0])
                    for i in range(bt):
                        path = predict_path[i]
                        tru_path = tags[i]
                        predict_tags = []
                        tru_tags = []
                        for j in range(s_l):
                            pre_tag = path[j]
                            tru_tag = tru_path[j]
                            predict_tags.append(ix2tag[pre_tag])
                            tru_tags.append(ix2tag[tru_tag])
                        predict_tags = predict_tags[:sent_length[i]]
                        tru_tags = tru_tags[:sent_length[i]]
                        batch_pre_path.append(predict_tags)
                        batch_tru_tag.append(tru_tags)

                    all_pre += batch_pre_path
                    all_tru_tag += batch_tru_tag
                    pbar.set_description("loss: %s" % (valid_loss))


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


