import pickle

import numpy as np
import torch
from .config import emb_path, emb_info_path


class WordEmbeddings:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.emb_path = emb_path
        self.emb_info_path = emb_info_path

        self.vocab = None
        self.word_to_int = None
        self.int_to_word = None
        self.emb = None

    @property
    def vocab_size(self):
        return len(self.vocab)

    def create_emb_matrix(self, vocab):
        self.vocab = vocab
        vocab_size = len(vocab)

        if self.emb_path is not None:
            pre_vectors, pre_word_to_int, pre_int_to_word = self.load_pretrained()
            word_to_vec = {w: pre_vectors[pre_word_to_int[w]] for w in pre_int_to_word}

            self.emb = np.zeros((vocab_size, self.embedding_dim))
            for i, word in enumerate(vocab):
                self.emb[i] = word_to_vec[word] if word in word_to_vec else np.random.normal(scale=0.6,
                                                                                             size=(self.embedding_dim,))
        else:
            self.emb = np.random.normal(scale=0.6, size=(vocab_size, self.embedding_dim))

        self.emb = torch.Tensor(self.emb)
        self.word_to_int = dict((w, i) for i, w in enumerate(vocab))
        self.int_to_word = dict((i, w) for i, w in enumerate(vocab))

    def load_pretrained(self):
        with open(self.emb_path, 'rb') as fin:
            vectors = np.load(fin)
        with open(self.emb_info_path, 'rb') as fin:
            data = pickle.load(fin)
            word_to_int = data['word_to_int']
            int_to_word = data['int_to_word']

        return vectors, word_to_int, int_to_word
