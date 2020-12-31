import collections
import numpy as np
from torch.utils.data import Dataset

START_TAG, END_TAG = "<START>", "<END>"
tag2ix = {"B-PER": 0, "I-PER": 1,
                  "B-LOC": 2, "I-LOC": 3,
                  "B-ORG": 4, "I-ORG": 5, "O": 6, START_TAG: 7, END_TAG: 8}
ix2tag = {v:k for k,v in tag2ix.items()}


class dataset(Dataset):
    def __init__(self,seq_length,mode):
        train_corpus_path = './data/train_corpus.txt'
        train_label_path = './data/train_label.txt'
        test_corpus_path = './data/test_corpus.txt'
        test_label_path = './data/test_label.txt'

        self.length = 0
        self.seq_length = seq_length
        train_corpus_splt = self.readline(train_corpus_path)
        self.voc2idx = self.build_vocab(train_corpus_splt)
        if mode == 'train':
            self.corpus_splt = train_corpus_splt
            self.label_splt = self.readline(train_label_path)
        if mode == 'test':
            self.corpus_splt = self.readline(test_corpus_path)
            self.label_splt = self.readline(test_label_path)

    def readline(self,path):
        res = []
        with open(path,'r',encoding='utf-8') as f:
            lines = f.readlines()
            self.length = len(lines)
            for line in lines:
                res.append(line.strip().split())
        return res

    def build_vocab(self,sents):
        word_counts = collections.Counter()
        for sent in sents:
            word_counts.update(sent)
        vocabulary =['<START>', '<UNK>', '<END>'] + [x[0] for x in word_counts]
        vocabulary_idx = {x:i for i,x in enumerate(vocabulary)}
        return vocabulary_idx

    def __getitem__(self, index):
        pad_words = ['，']
        pad_label = ['O']
        words = self.corpus_splt[index]
        sent_length = len(words)
        words = words + pad_words * self.seq_length
        tags = self.label_splt[index] + pad_label * self.seq_length # 加start_tag是为了给转移概率初始矩阵
        words = words[:self.seq_length]
        tags = tags[:self.seq_length]
        words = np.asarray([self.voc2idx.get(w, 1) for w in words])  #转为numpy很重要，如果是list就会导致，不会自动转为tensor
        tags = np.asarray([tag2ix[t] for t in tags])
        return words,tags,sent_length

    def __len__(self):
        return self.length
