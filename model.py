import torch
from torch import nn
from config import *


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(smat):
    vmax = smat.max(dim=-2, keepdim=True).values  # 每一列的最大数,考虑第一维是batch_size,dim取-2
    return (smat - vmax).exp().sum(axis=-2, keepdim=True).log() + vmax


class BiLSTM_CRF(nn.Module):
    def __init__(self, pos2id, word2idx, device):
        super(BiLSTM_CRF, self).__init__()
        self.device = device
        self.pos2id = pos2id
        self.target_size = len(pos2id)
        self.embed = nn.Embedding(len(word2idx), args.word_dim)
        self.rnn = nn.LSTM(input_size=args.word_dim, hidden_size=args.hidden_num // 2, batch_first=True,
                           bidirectional=True)
        self.hidden2tag = nn.Linear(args.hidden_num, self.target_size)
        # 转移矩阵(i, j) to j from i
        self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size).to(self.device),
                                        requires_grad=True)
        # 开始和结束的转移限制
        self.transitions.data[pos2id['STOP'], :] = -10000
        self.transitions.data[:, pos2id['START']] = -10000

    # 计算所有路径的得分
    def _forward_alg(self, feats):  # feats:shape(batch_size, seq_length, target_size)
        init_alphas = torch.full((len(feats), 1, self.target_size), -10000.).to(self.device)  # shape(
        # batch_size, 1 , target_size)
        init_alphas[:, 0, self.pos2id['START']] = 0
        feats = feats.permute(1, 0, 2)  # shape(seq_length, batch_size, target_size)
        for feat in feats:  # feat:shape(batch_size, target_size)
            # shape(batch_size, target_size, 1)
            # shape(batch_size, 1, target_size)
            # shape(target_size, target_size)  broadcast
            init_alphas = log_sum_exp(init_alphas.transpose(1, 2) + feat.unsqueeze(1) + self.transitions)  # shape(
            # batch_size, 1, target_size)
        # 最后转到EOS，发射分值为0，转移分值为列向量 self.transitions[:, [self.tag2ix[END_TAG]]]
        return log_sum_exp(init_alphas.transpose(1, 2) + self.transitions[:, [self.pos2id['STOP']]]) \
            .flatten().sum()  # 把最后一步转移加上

    def _get_lstm_features(self, sentence, seq_length):
        embeds = self.embed(sentence)  # shape(batch_size, seq_length, word_dim)
        lstm_out, _hidden = self.rnn(embeds)
        #lstm_out, (h, c) = self.lstm(embeds, seq_length)  # shape(batch_size, seq_length, hidden_num)
        lstm_feats = self.hidden2tag(lstm_out)  # shape(batch_size, seq_length, target_size)
        return lstm_feats

    def _score_sentence(self, feats, tags):  # tags:shape(batch_size, seq_length)
        score = torch.zeros(1).to(self.device)
        start = torch.tensor(self.pos2id['START']).unsqueeze(-1).repeat(len(feats), 1).to(
            self.device)  # shape(# batch_size, 1)
        tags_new = torch.cat((start, tags), dim=-1)  # shape(batch_size, seq_length + 1)
        feats_all = torch.gather(feats, dim=-1, index=tags.unsqueeze(-1)).sum(dim=1).sum()
        transition_all = self.transitions[tags_new[:, :-1].flatten(), tags_new[:, 1:].flatten()].sum()
        score = score + transition_all + feats_all
        stop = torch.full((len(feats),), self.pos2id['STOP'], dtype=torch.long)
        score_2end = self.transitions[tags_new[:, -1], stop].sum()
        score = score + score_2end
        return score

    def neg_log_likelihood(self, words, tags, seq_length):  # 求一对 <sentence, tags> 在当前参数下的负对数似然，作为loss
        feats = self._get_lstm_features(words, seq_length)
        gold_score = self._score_sentence(feats, tags)
        forward_score = self._forward_alg(feats)
        batch_loss = (forward_score - gold_score) / len(feats)
        return batch_loss

    def _viterbi_decode(self, feats):
        backtrace = []  # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        alpha = torch.full((len(feats), 1, self.target_size), -10000.).to(self.device)
        alpha[:, 0, self.pos2id['START']] = 0
        feats = feats.permute(1, 0, 2)
        # 这里跟计算所有路径得分类似但又有点不同
        for feat in feats:  # feat:shape(batch_size, target_size)
            alpha = alpha.transpose(1, 2) + self.transitions  # shape(batch_size, target_size, target_size)
            best_tag_id = alpha.argmax(dim=1)  # shape(batch_size, target_size)
            backtrace.append(best_tag_id)
            viterbi_var, _ = alpha.max(dim=1)  # shape(batch_size, target_size)
            alpha = (viterbi_var + feat).unsqueeze(1)  # shape(batch_size, 1, target_size)
        alpha = alpha.transpose(1, 2) + self.transitions[:, [self.pos2id['STOP']]]  # shape(batch_size, target_size, 1)
        best_tag_id = alpha.argmax(dim=1)  # shape(batch_size, 1)
        path_score, _ = alpha.max(dim=1)  # shape(batch_size, 1)
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace):  # bptrs_t:shape(batch_size, target_size)
            best_tag_id = torch.gather(bptrs_t, 1, best_tag_id)  # shape(batch_size, 1)
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert all(start == self.pos2id['START'])
        best_path.reverse()
        return torch.cat(best_path, dim=1)

    def forward(self, words, seq_length):
        lstm_feats = self._get_lstm_features(words, seq_length)
        tag_seq = self._viterbi_decode(lstm_feats)
        return tag_seq
