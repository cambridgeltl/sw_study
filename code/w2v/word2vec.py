# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  31/05/2018 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import time
import sys

import torch

sys.path.append('../scripts')
import ios, nn_utils
from args import create_w2v_parser, parse_args
from w2v_model import Word2Vec

import pdb

np.random.seed(1234)
torch.manual_seed(1234)


def train(m):
  t_start = time.monotonic()
  # actual learning rate
  lr = 0
  # temporary word count
  word_ct = 0 
  # total word count
  word_tot_count = 0
  # pairs for batch training
  pairs = []
  # numbers of batches
  bs_n = 0
  # total loss for a checkpoint
  total_loss = .0
  # optimizer
  optimizer = nn_utils.get_optimizer(m)

  for i in range(m.iters):
    with open(m.data.infile, 'r') as fin:
      for line in fin:
        word_ct = get_pairs(m, pairs, line, word_ct)
        if len(pairs) < m.bs:
          # not engouh training pairs
          continue

        total_loss += train_batch(m, optimizer, pairs[:m.bs], m.bs) 
        pairs = pairs[m.bs:]
        bs_n += 1

        if word_ct > 1e4:
          word_tot_count += word_ct
          word_ct = 0
          prog = cur_count / (m.iters * m.data.word_ct)
          lr = nn_utils.schedule_lr(m, optimizer, prog)
          ios.output_states(lr, prog, word_tot_count, m, total_loss, bs_n, t_start) 
          total_loss = .0
          bs_n = 0
       
  if pairs:
    while pairs:
      total_loss += train_batch(m, optimizer, pairs[:m.bs], len(pairs[:m.bs]))
      pairs = pairs[m.bs:]
      bs_n += 1
    word_tot_count += word_ct
    prog = cur_count / (m.iters * m.data.word_ct)
    ios.output_states(lr, prog, word_tot_count, m, total_loss, bs_n, t_start) 

  print('\nOutput to file: {}\nSave model to: {}'.format(m.outfile, m.save_model)) 
  m.model.save_embedding(m)


def get_pairs(m, pairs, line, word_ct):
    linevec_idx = [m.data.word2idx[w] for w in line.strip().split() if w in m.data.word2idx]
    word_ct += len(linevec_idx)
    # subsampling
    linevec_idx = [w_idx for w_idx in linevec_idx if np.random.random_sample() <= m.sub_samp_probs[w_idx]]
    if len(linevec_idx) > 1:
      # sentence has more than 1 word
      pairs += m.data.get_batch_pairs(linevec_idx, m.win_size)
    return word_ct


def get_progress(m, cur_count):
  prog = cur_count / (m.iters * m.data.word_ct)
  return prog


def train_batch(m, optimizer, pairs, bs):
  pos_u, pos_v, neg_v = gen_batch(m, pairs, bs)

  optimizer.zero_grad()
  loss = m.model(pos_u, pos_v, neg_v, m)
  loss.backward()
  # gradient clipping
  if m.gclip > 0:
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.grad is not None and p.requires_grad and not p.grad.data.is_sparse, m.model.parameters()), m.gclip)
  optimizer.step()
  return loss.item()


def gen_batch(m, pairs, bs):
  pos_u, pos_v = map(list, zip(*pairs))
  neg_v = np.random.choice(m.neg_idxs, (bs, m.neg_n), p = m.data.neg_sample_probs)
  return (pos_u, pos_v, neg_v)


if __name__ == '__main__':
  parser = create_w2v_parser()
  args = parse_args(parser)

  w2v = Word2Vec(args)
  w2v.init_model(args)

  train(w2v) 
