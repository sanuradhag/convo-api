import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import nltk
import torch.nn.init as init

from .helpers import *
from .config import *


class DataLoader:
    def __init__(self, word_emb, gpu=False, mode='train'):
        self.filepath = dataset_path
        self.word_emb = word_emb
        self.gpu = gpu
        self.start_token_str = START_TOKEN
        self.end_token_str = END_TOKEN
        self.pad_token_str = PAD_TOKEN

        self.train_size = TRAIN_SIZE
        self.test_size = TEST_SIZE
        self.total_samples = self.train_size + self.test_size
        self.mode = mode

        self.start_token = 2
        self.end_token = 1
        self.pad_token = 0
        self.cond_samples = None
        self.pos_samples = None
        self.cond_lens = None
        self.pos_lens = None
        self.word_to_int = None
        self.int_to_word = None
        self.vocab = None
        self.max_seq_len = 0

        self.fetcher = self.fetch()
        self.frozen = None 

    def load(self):
        cond_samples = []
        pos_samples = []
        vocab = [self.start_token_str, self.end_token_str, self.pad_token_str]
        with open(self.filepath, 'r', encoding='utf-8') as fin:
            fin.readline() # ignore header
            reader = csv.reader(fin, delimiter='\t')

            for row in reader:
                if len(cond_samples) >= self.total_samples:
                    break

                # columns: id, qid1, qid2, question1, question2, is_duplicate
                if row[-1] == '1':
                    q1_str = row[3].lower()
                    q2_str = row[4].lower()
                    q1_tokens = nltk.word_tokenize(q1_str)
                    q2_tokens = nltk.word_tokenize(q2_str)

                    q1 = [self.start_token_str] + q1_tokens + [self.end_token_str]
                    q2 = [self.start_token_str] + q2_tokens + [self.end_token_str]
                    cond_samples.append(q1)
                    pos_samples.append(q2)
                    vocab += q1_tokens + q2_tokens

        self.vocab = sorted(list(set(vocab)))
        for i in range (29):
            ss = 'x' + str(i)
            self.vocab.append(ss)
        self.word_emb.create_emb_matrix(self.vocab)
        self.word_to_int, self.int_to_word = self.word_emb.word_to_int, self.word_emb.int_to_word

        # Keep only train/test set
        start, end = (0, self.train_size) if self.mode == 'train' else (self.train_size, self.train_size + self.test_size)

        # Map dataset
        self.cond_samples = [self.sent_to_ints(q) for q in cond_samples[start:end]]
        self.pos_samples = [self.sent_to_ints(q) for q in pos_samples[start:end]]

        # Map special tokens
        self.start_token = self.word_to_int[self.start_token_str]
        self.end_token = self.word_to_int[self.end_token_str]
        self.pad_token = self.word_to_int[self.pad_token_str]
        
        # Pad dataset, turn into np array
        self.cond_samples, self.cond_lens = pad_samples(self.cond_samples, self.pad_token)
        self.pos_samples, self.pos_lens = pad_samples(self.pos_samples, self.pad_token)
        self.max_seq_len = max(torch.max(self.cond_lens).item(), torch.max(self.pos_lens).item())

    def sample(self, num_samples, is_val=False, gpu=False):
        # Fetch next batch
        next(self.fetcher)
        pos_samples, pos_lens, cond_ids, end_of_dataset = self.fetcher.send(num_samples)

        # Freeze samples if is validation set
        if is_val:
            self.freeze(cond_ids[0], cond_ids[-1])

        # Put to GPU
        if gpu:
            pos_samples = pos_samples.cuda()
            pos_lens = pos_lens.cuda()

        # Trim
        pos_samples = trim_trailing_paddings(pos_samples, pos_lens)

        return pos_samples, pos_lens, cond_ids, end_of_dataset

    def fetch_cond_samples(self, cond_ids, gpu=False):
        cond_samples, cond_lens = self.cond_samples[cond_ids], self.cond_lens[cond_ids]

        # Put to GPU
        if gpu:
            cond_samples = cond_samples.cuda()
            cond_lens = cond_lens.cuda()
        
        # Trim
        cond_samples = trim_trailing_paddings(cond_samples, cond_lens)

        return cond_samples, cond_lens

    def sent_to_ints(self, s):
        ints = []
        for w in s:
            if w in self.word_to_int:
                ints += [self.word_to_int[w]]
        return ints

    def ints_to_sent(self, ints):
        return ' '.join([self.int_to_word[i] for i in ints])

    def fetch(self):
        i = 0
        n = len(self.cond_samples)
        while i < n:
            num_samples = yield
            j = min(i + num_samples, n)
            sample_idx = list(range(i, j))
            
            # Check if overlap with frozen batch
            if self.frozen is not None:
                start, end = self.frozen
                if i < start and j > start:
                    remaining_after_end = num_samples - (start - i)
                    j = min(end + 1 + remaining_after_end, n)
                    sample_idx = list(range(i, start)) + list(range(end + 1, j))
                elif i >= start and i <= end:
                    j = min(end + 1 + num_samples, n)
                    sample_idx = list(range(end + 1, j))

            # Fetch
            yield (self.pos_samples[sample_idx], self.pos_lens[sample_idx], sample_idx, j == n)

            # Move pointer
            if j == n: # reset
                i = 0
                self.shuffle()
            else: # continue
                i = j

    def shuffle(self):
        num_samples = self.pos_samples.shape[0]
        
        if self.frozen is not None:
            start, end = self.frozen
            perm = torch.cat([torch.randperm(start), \
                              torch.LongTensor(list(range(start, end + 1))), \
                              (torch.randperm(num_samples - end - 1) + end + 1)])
        else:
            perm = torch.randperm(num_samples)

        self.pos_samples = self.pos_samples[perm]
        self.pos_lens = self.pos_lens[perm]
        self.cond_samples = self.cond_samples[perm]
        self.cond_lens = self.cond_lens[perm]

    def freeze(self, start, end):
        self.frozen = (start, end) 

    def release(self):
        self.frozen = None
    
    def reset(self):
        self.fetcher = self.fetch()
        self.frozen = None
        self.shuffle()
