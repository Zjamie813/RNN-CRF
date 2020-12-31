#encoding:utf-8

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, "input.en.txt")
        vocab_file = os.path.join(data_dir, "vocab.en.pkl")

        self.preprocess(input_file, vocab_file)
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        word_counts = collections.Counter()  # 计数器，词频
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent)
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq] # x[1]是词频，x[0]是词
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} # 包括k：词，v:索引
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file):
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            if lines[0][:1] == codecs.BOM_UTF8:
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]

        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)
        #print 'word num: ', self.vocab_size

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.vocab, f)

        raw_data = [[0] * self.seq_length +
            [self.vocab.get(w, 1) for w in line] +
            [2] * self.seq_length for line in lines] # get(w,1)代表不在词表里的词就用1代替
        self.raw_data = raw_data

    def create_batches(self):
        xdata, ydata = list(), list()
        for row in self.raw_data: # 每一行就是一句话
            for ind in range(self.seq_length, len(row)):
                xdata.append(row[ind-self.seq_length:ind]) # 从ind往前选seq_length个单词作为x
                ydata.append([row[ind]]) # 紧接着x的单词为y
        self.num_batches = int(len(xdata) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(xdata[:self.num_batches * self.batch_size]) # 每一个元素都是一个训练样本，可以是从一个句子里面提出来也可以不是
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches, 0) # 平均切分
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
