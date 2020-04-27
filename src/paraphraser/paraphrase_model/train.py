from __future__ import print_function
import argparse
import logging
import time
from math import ceil
import sys
import os
import pdb

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from .generator import Generator
from .discriminator import Discriminator
from ..word_embeddings import WordEmbeddings
from ..dataloader import DataLoader
from ..config import *


def train_generator_MLE(gen, gen_opt, data_loader, epochs, save_path):
    gen.train()
    data_loader.reset()

    teacher_forcing_ratio = TEACHER_FORCING_RATIO
    for epoch in range(epochs):
        print(f'epoch {epoch + 1} : ', end='')

        total_loss = 0
        i = 0
        end_of_dataset = False
        while not end_of_dataset: 
            # Sample from data_loader
            pos_samples, pos_lens, cond_ids, end_of_dataset = data_loader.sample(BATCH_SIZE, gpu=CUDA)
            cond_samples, cond_lens = data_loader.fetch_cond_samples(cond_ids, gpu=CUDA)

            # Train
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(cond_samples, cond_lens, pos_samples, pos_lens, teacher_forcing_ratio=teacher_forcing_ratio, gpu=CUDA)
            loss.backward()
            gen_opt.step()

            # Accumulate loss
            total_loss += loss.item() * len(pos_samples)

            # Log
            if i % ceil(ceil(data_loader.total_samples / float(BATCH_SIZE)) / 10.) == 0:
                print('.', end='')
                sys.stdout.flush()

            i += 1

        total_loss /= data_loader.total_samples
        logging.info(f'[G_MLE] epoch = {epoch + 1}, average_train_NLL = {total_loss:.4f}')

        if epoch != 0 and epoch % TEACHER_FORCING_UPDATE_EP == 0:
            teacher_forcing_ratio -= TEACHER_FORCING_RATIO_DECR_STEP
            teacher_forcing_ratio = max(teacher_forcing_ratio, 0)

    
    torch.save(gen.state_dict(), save_path)

def train_generator_PG(gen, gen_opt, dis, data_loader, rollout, g_steps, adv_iter, save_path):
    gen.train()
    gen.turn_on_grads()
    dis.turn_off_grads()
    data_loader.reset()
    total_loss = 0
    i = 0
    end_of_dataset = False
    total_samples = 0
    for g_step in range(g_steps):
        if end_of_dataset:
            break
        rollout.load_state_dict(gen.state_dict())
        target, target_lens, pos_samples, pos_lens, cond_samples, cond_lens, end_of_dataset \
                = helpers.prepare_generator_batch(data_loader, gen, BATCH_SIZE, gpu=CUDA)
        rollout_targets, rollout_target_lens, rollout_cond, rollout_cond_lens \
                = rollout.rollout(target, target_lens, cond_samples, cond_lens, ROLLOUT_NUM, gpu=CUDA)
        rollout_cond_shape = rollout_cond.shape
        rollout_targets_shape = rollout_targets.shape

        rollout_rewards = dis.batchClassify(
                              rollout_targets.view(-1, rollout_targets_shape[-1]),
                              rollout_target_lens.view(-1),
                              rollout_cond.view(-1, rollout_cond_shape[-1]),
                              rollout_cond_lens.view(-1)
                          ).view(rollout_targets_shape[:-1])
        rollout_rewards = torch.mean(rollout_rewards, -1)
        rewards = dis.batchClassify(target, target_lens, cond_samples, cond_lens).unsqueeze(0)
        total_rewards = torch.cat([rollout_rewards, rewards])
        gen_opt.zero_grad()
        loss = gen.batchPGLoss(cond_samples, target, total_rewards, gpu=CUDA)
        loss.backward()
        gen_opt.step()
        total_loss += loss.item() * len(target)
        gen_opt.zero_grad()
        loss = gen.batchNLLLoss(cond_samples, cond_lens, pos_samples, pos_lens, teacher_forcing_ratio=1, gpu=CUDA)
        loss.backward()
        gen_opt.step()

        total_samples += len(target)
        i += 1

    if total_samples != 0:
        total_loss = total_loss / total_samples
    else:
        total_loss = float('nan')
        
    logging.info(f'[G_PG] iter = {adv_iter}, average_train_NLL = {total_loss:.4f}')
    torch.save(gen.state_dict(), save_path)

def train_discriminator(dis, dis_opt, gen, data_loader, d_steps, epochs, adv_iter, save_path):
    dis.train()
    dis.turn_on_grads()
    gen.turn_off_grads()
    data_loader.reset()
    valid_set_size = int(data_loader.train_size * VALID_SET_SIZE_RATIO) * 2
    valid_set_size -= valid_set_size % BATCH_SIZE # align with batch size
    train_set_size = int(((data_loader.train_size * 2) - valid_set_size) / 2)

    val_inp, val_inp_lens, val_cond, val_cond_lens, val_target, end_of_dataset \
            = helpers.prepare_discriminator_data(data_loader, gen, valid_set_size, is_val=True, on_cpu=True, gpu=CUDA, gpu_limit=BATCH_SIZE)
    for d_step in range(d_steps):
        all_inp, all_inp_lens, all_cond, all_cond_lens, all_target, end_of_dataset \
                = helpers.prepare_discriminator_data(data_loader, gen, train_set_size, on_cpu=True, gpu=CUDA, gpu_limit=BATCH_SIZE)
        for epoch in range(epochs):
            print(f'd-step {d_step + 1} epoch {epoch + 1} : ', end='')

            total_loss = 0
            total_acc = 0
            for i in range(0, train_set_size, BATCH_SIZE):
                inp, inp_lens, cond, cond_lens, target \
                        = all_inp[i:i+BATCH_SIZE], all_inp_lens[i:i+BATCH_SIZE], \
                          all_cond[i:i+BATCH_SIZE], all_cond_lens[i:i+BATCH_SIZE], all_target[i:i+BATCH_SIZE]

                if CUDA:
                    inp, inp_lens, cond, cond_lens, target = inp.cuda(), inp_lens.cuda(), cond.cuda(), cond_lens.cuda(), target.cuda()
                dis_opt.zero_grad()
                loss, acc = dis.batchBCELoss(inp, inp_lens, cond, cond_lens, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.item() * len(inp)
                total_acc += acc

                if (i / BATCH_SIZE) % ceil(ceil(data_loader.total_samples / float(BATCH_SIZE / 2)) / 10.) == 0: # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            if i != 0:
                total_loss /= train_set_size 
                total_acc /= train_set_size
                dis.eval()
                if CUDA:
                    val_inp, val_inp_lens, val_cond, val_cond_lens, val_target = val_inp.cuda(), val_inp_lens.cuda(), val_cond.cuda(), val_cond_lens.cuda(), val_target.cuda()

                with torch.no_grad():
                    val_acc = 0
                    for i in range(0, valid_set_size, BATCH_SIZE):
                        _, acc = dis.batchBCELoss(val_inp[i:i+BATCH_SIZE], val_inp_lens[i:i+BATCH_SIZE], val_cond[i:i+BATCH_SIZE],
                                                  val_cond_lens[i:i+BATCH_SIZE], val_target[i:i+BATCH_SIZE])
                        val_acc += acc
                    val_acc /= valid_set_size

                val_inp, val_inp_lens, val_cond, val_cond_lens, val_target = val_inp.cpu(), val_inp_lens.cpu(), val_cond.cpu(), val_cond_lens.cpu(), val_target.cpu()
                dis.train()

                logging.info(f'[D] iter = {adv_iter}, step = {d_step}, epoch = {epoch+1}, average_loss = {total_loss:.4f}, train_acc = {total_acc:.4f}, val_acc = {val_acc:.4f}')

            end_of_dataset = False

        torch.save(dis.state_dict(), save_path)

    torch.save(dis.state_dict(), save_path)
    data_loader.release()

# MAIN
if __name__ == '__main__':
    t = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    word_emb = word_embeddings.WordEmbeddings(ED, pretrained_emb_path_prefix)
    data_loader = dataloader.DataLoader(dataset_path, word_emb, train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                                   start_token_str=START_TOKEN, end_token_str=END_TOKEN, pad_token_str=PAD_TOKEN,
                                   gpu=CUDA, light_ver=LIGHT_VER)
    data_loader.load()
    start_token, end_token, pad_token, max_seq_len = data_loader.start_token, data_loader.end_token, data_loader.pad_token, data_loader.max_seq_len
    max_seq_len += MAX_SEQ_LEN_PADDING

    gen = generator.Generator(ED, G_HD, word_emb, start_token=start_token, end_token=end_token, pad_token=pad_token,
                              max_seq_len=max_seq_len, gpu=CUDA)
    dis = discriminator.Discriminator(ED, D_HD, word_emb, start_token=start_token, end_token=end_token, pad_token=pad_token,
                                      max_seq_len=max_seq_len, gpu=CUDA)
    rollout = generator.Generator(ED, G_HD, word_emb, start_token=start_token, end_token=end_token, pad_token=pad_token,
                                  max_seq_len=max_seq_len, gpu=CUDA)
    rollout.turn_off_grads()


    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,filename='../log/convo-log.log')


    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    dis_optimizer = optim.Adagrad(dis.parameters())

    train_generator_MLE(gen, gen_optimizer, data_loader, G_PRETRAIN_EPOCHS, pretrain_gen_path)
    train_discriminator(dis, dis_optimizer, gen, data_loader, D_PRETRAIN_STEPS, D_PRETRAIN_EPOCHS, -1, pretrain_dis_path)


    for i in range(ADV_TRAIN_ITERS):
        train_generator_PG(gen, gen_optimizer, dis, data_loader, rollout, G_TRAIN_STEPS, iteration, gen_model_path)
        train_discriminator(dis, dis_optimizer, gen, data_loader, D_TRAIN_STEPS, D_TRAIN_EPOCHS, iteration, dis_model_path)
