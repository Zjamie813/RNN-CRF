#!user/bin/env python
# -*- coding:utf-8 -*-
# author: DingYang time:2020/10/19
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max_seq_length', type=int, default=100,
                    help='max_seq_length')
parser.add_argument('--batch_size', type=int, default=512,
                    help='minibatch size')
parser.add_argument('--hidden_num', type=int, default=512,
                    help='number of hidden layers')
parser.add_argument('--word_dim', type=int, default=100,
                    help='number of word embedding')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--interval_save', type=float, default=1,
                    help='clip gradients at this value')

args = parser.parse_args()  # 参数集合
