import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class CRFLayer(nn.Module):
    NEG_LOGIT = -100000.
    def __init__(self, word2ix,embedding_dim,hidden_size, num_entity_labels):
        super(CRFLayer, self).__init__()

        self.tag_size = num_entity_labels + 2  # add start tag and end tag
        self.start_tag = self.tag_size - 2 # start sign label idx
        self.end_tag = self.tag_size - 1 # end sign label idx

        self.word_embeds = nn.Embedding(len(word2ix), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size // 2, num_layers=2, bidirectional=True)

        # Map token-level hidden state into tag scores
        self.hidden2tag = nn.Linear(hidden_size, self.tag_size)
        # Transition Matrix
        # [i, j] denotes transitioning from j to i
        # [tag_size, tag_size]
        self.trans_mat = nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.reset_trans_mat()

    def reset_trans_mat(self):
        nn.init.kaiming_uniform_(self.trans_mat, a=math.sqrt(5))  # copy from Linear init
        # set parameters that will not be updated during training, but is important
        self.trans_mat.data[self.start_tag, :] = self.NEG_LOGIT
        self.trans_mat.data[:, self.end_tag] = self.NEG_LOGIT

    def get_log_parition(self, seq_emit_score):
        """
        Calculate the log of the partition function
        :param seq_emit_score: [seq_len, batch_size, tag_size]
        :return: Tensor with Size([batch_size])
        """
        seq_len, batch_size, tag_size = seq_emit_score.size()
        # dynamic programming table to store previously summarized tag logits
        # dp_table [batch_size, tag_size]
        dp_table = seq_emit_score.new_full(
            (batch_size, tag_size), self.NEG_LOGIT, requires_grad=False
        )
        dp_table[:, self.start_tag] = 0.

        batch_trans_mat = self.trans_mat.unsqueeze(0).expand(batch_size, tag_size, tag_size)

        for token_idx in range(seq_len):
            prev_logit = dp_table.unsqueeze(1)  # [batch_size, 1, tag_size]
            batch_emit_score = seq_emit_score[token_idx].unsqueeze(-1)  # [batch_size, tag_size, 1]
            cur_logit = batch_trans_mat + batch_emit_score + prev_logit  # [batch_size, tag_size, tag_size]
            dp_table = log_sum_exp(cur_logit)  # [batch_size, tag_size]
        batch_logit = dp_table + self.trans_mat[self.end_tag, :].unsqueeze(0)
        log_partition = log_sum_exp(batch_logit)  # [batch_size]

        return log_partition

    def get_gold_score(self, seq_emit_score, seq_token_label):
        """
        Calculate the score of the given sequence label
        :param seq_emit_score: [seq_len, batch_size, tag_size]
        :param seq_token_label: [seq_len, batch_size]
        :return: Tensor with Size([batch_size])
        """
        seq_len, batch_size, tag_size = seq_emit_score.size()

        # end_token_label [1, batch_size] (end_tag)
        end_token_label = seq_token_label.new_full((1, batch_size), self.end_tag, requires_grad=False)
        # seq_cur_label : [seq_len+1, batch_size] - > [seq_len+1, batch_size, 1, 1] - > [seq_len+1, batch_size, 1, tag_size]
        # seq_cur_label :  [seq_len+1, batch_size, 1, tag_size]
        #
        seq_cur_label = torch.cat([seq_token_label, end_token_label], dim=0).unsqueeze(-1).unsqueeze(-1).expand(seq_len + 1, batch_size, 1, tag_size)

        # start_token_label [1, batch_size] (start_tag)
        start_token_label = seq_token_label.new_full((1, batch_size), self.start_tag, requires_grad=False)

        seq_prev_label = torch.cat(
            [start_token_label, seq_token_label], dim=0
        ).unsqueeze(-1).unsqueeze(-1)  # [seq_len+1, batch_size, 1, 1]

        # [seq_len+1, batch_size, tag_size, tag_size]
        seq_trans_score = self.trans_mat.unsqueeze(0).unsqueeze(0).expand(seq_len + 1, batch_size, tag_size, tag_size)
        # gather according to token label at the current token
        # 得到标准路径的score
        gold_trans_score = torch.gather(seq_trans_score, 2, seq_cur_label)  # [seq_len+1, batch_size, 1, tag_size]
        # gather according to token label at the previous token
        #
        gold_trans_score = torch.gather(gold_trans_score, 3, seq_prev_label)  # [seq_len+1, batch_size, 1, 1]

        batch_trans_score = gold_trans_score.sum(dim=0).squeeze(-1).squeeze(-1)  # [batch_size]

        gold_emit_score = torch.gather(seq_emit_score, 2, seq_token_label.unsqueeze(-1))  # [seq_len, batch_size, 1]
        batch_emit_score = gold_emit_score.sum(dim=0).squeeze(-1)  # [batch_size]

        gold_score = batch_trans_score + batch_emit_score  # [batch_size]

        return gold_score

    def viterbi_decode(self, seq_emit_score):
        """
        Use viterbi decoding to get prediction
        :param seq_emit_score: [seq_len, batch_size, tag_size]
        :return:
            batch_best_path: [batch_size, seq_len], the best tag for each token
            batch_best_score: [batch_size], the corresponding score for each path
        """
        seq_len, batch_size, tag_size = seq_emit_score.size()
        # db_table [batch_size, tag_size]
        dp_table = seq_emit_score.new_full((batch_size, tag_size), self.NEG_LOGIT, requires_grad=False)
        dp_table[:, self.start_tag] = 0
        backpointers = []

        for token_idx in range(seq_len):
            last_tag_score = dp_table.unsqueeze(-2)  # [batch_size, 1, tag_size]
            batch_trans_mat = self.trans_mat.unsqueeze(0).expand(batch_size, tag_size, tag_size)
            cur_emit_score = seq_emit_score[token_idx].unsqueeze(-1)  # [batch_size, tag_size, 1]
            cur_trans_score = batch_trans_mat + last_tag_score + cur_emit_score  # [batch_size, tag_size, tag_size]
            dp_table, cur_tag_bp = cur_trans_score.max(dim=-1)  # [batch_size, tag_size]
            backpointers.append(cur_tag_bp)
        # transition to the end tag
        last_trans_arr = self.trans_mat[self.end_tag].unsqueeze(0).expand(batch_size, tag_size)
        dp_table = dp_table + last_trans_arr

        # get the best path score and the best tag of the last token
        batch_best_score, best_tag = dp_table.max(dim=-1)  # [batch_size]
        best_tag = best_tag.unsqueeze(-1)  # [batch_size, 1]
        best_tag_list = [best_tag]
        # reversely traverse back pointers to recover the best path
        for last_tag_bp in reversed(backpointers):
            # best_tag Size([batch_size, 1]) records the current tag that can own the highest score
            # last_tag_bp Size([batch_size, tag_size]) records the last best tag that the current tag is based on
            best_tag = torch.gather(last_tag_bp, 1, best_tag)  # [batch_size, 1]
            best_tag_list.append(best_tag)
        batch_start = best_tag_list.pop()
        # print('(batch_start == self.start_tag).sum().item():{}'.format((batch_start == self.start_tag).sum().item()))
        # print('batch_size:{}'.format(batch_size))

        assert (batch_start == self.start_tag).sum().item() == batch_size
        best_tag_list.reverse()
        batch_best_path = torch.cat(best_tag_list, dim=-1)  # [batch_size, seq_len]

        return batch_best_path, batch_best_score

    def _get_lstm_features(self, words):  # 求出每一帧对应的隐向量
        # LSTM输入形状(seq_len, batch=1, input_size); 教学演示 batch size 为1
        embeds = self.word_embeds(words) # [bt,seq_length,dim]
        embeds = embeds.permute(1,0,2)
        # 随机初始化LSTM的隐状态H
        #hidden = torch.randn(2, batch_size, self.hidden_dim // 2), torch.randn(2, batch_size, self.hidden_dim // 2)
        lstm_out, _hidden = self.lstm(embeds)  # lstm_out:[seq_len,batch,hidden_dim],是hidden维度的输出,hidden:两个[2,1,2]
        return lstm_out  # 把LSTM输出的隐状态张量去掉batch维，然后降维到tag空间

    def forward(self, words, seq_token_label=None, train_flag=True, decode_flag=True):
        """
        Get loss and prediction with CRF support.
        :param words: assume size [batch_size,seq_len,hidden_size]
        :param seq_token_label: assume size [batch_size,seq_len]
        :param batch_first: Flag to denote the meaning of the first dimension
        :param train_flag: whether to calculate the loss
        :param decode_flag: whether to decode the path based on current parameters
        :return:
            nll_loss: negative log-likelihood loss
            seq_token_pred: seqeunce predictions
        """
        seq_token_emb = self._get_lstm_features(words) #[seq,bt,hiddensize]
        seq_token_label = seq_token_label.transpose(0, 1).contiguous()
        # if batch_first:
        #     # CRF assumes the input size of [seq_len, batch_size, hidden_size]
        #     seq_token_emb = seq_token_emb.transpose(0, 1).contiguous()
        #     if seq_token_label is not None:
        #         seq_token_label = seq_token_label.transpose(0, 1).contiguous()

        seq_emit_score = self.hidden2tag(seq_token_emb)  # [seq_len, batch_size, tag_size]
        if train_flag:
            gold_score = self.get_gold_score(seq_emit_score, seq_token_label)  # [batch_size]
            log_partition = self.get_log_parition(seq_emit_score)  # [batch_size]
            nll_loss = (log_partition - gold_score).mean()
        else:
            nll_loss = None

        if decode_flag:
            # Use viterbi decoding to get the current prediction
            # no matter what batch_first is, return size is [batch_size, seq_len]
            batch_best_path, batch_best_score = self.viterbi_decode(seq_emit_score)
        else:
            batch_best_path = None

        return nll_loss, batch_best_path

    # Compute log sum exp in a numerically stable way


def log_sum_exp(batch_logit):
    """
    Caculate the log-sum-exp operation for the last dimension.
    :param batch_logit: Size([*, logit_size]), * should at least be 1
    :return: Size([*])
    """
    batch_max, _ = batch_logit.max(dim=-1)
    batch_broadcast = batch_max.unsqueeze(-1)
    return batch_max + \
           torch.log(torch.sum(torch.exp(batch_logit - batch_broadcast), dim=-1))
