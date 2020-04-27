import argparse

import torch
import nltk

from .paraphrase_model.generator import Generator
from .word_embeddings import WordEmbeddings
from .dataloader import DataLoader

from .config import *


class Paraphraser():
    def __init__(self):

        word_emb = WordEmbeddings(ED)
        self.data_loader = DataLoader(word_emb, gpu=False, mode='train')
        self.data_loader.load()

        end_token, pad_token, max_seq_len, vocab_size = self.data_loader.end_token, self.data_loader.pad_token, self.data_loader.max_seq_len, len(self.data_loader.vocab)
        max_seq_len += MAX_SEQ_LEN_PADDING

        self.generator = Generator(ED, G_HD, word_emb,end_token=end_token, pad_token=pad_token, max_seq_len=max_seq_len, gpu=False)

        self.generator.load_state_dict(torch.load(gen_model_path,  map_location=torch.device('cpu')))

        self.generator.eval()
        self.generator.turn_off_grads()

    def tensor_to_sent(self,t):
        ints = t.numpy()
        ints = [x for x in ints if x != self.data_loader.pad_token]
        sent = self.data_loader.ints_to_sent(ints)
        sent = sent.replace(self.data_loader.end_token_str, '').replace(self.data_loader.start_token_str, '').strip()
        return sent

    def generate_single(self,sentence):
        sentence_tokens = nltk.word_tokenize(sentence.lower()) + [self.data_loader.end_token_str]
        cond = torch.LongTensor(self.data_loader.sent_to_ints(sentence_tokens))

        result = self.generator.sample_until_end(cond, max_len=5)
        result_str = self.tensor_to_sent(result)

        return result_str



